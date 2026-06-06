# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the DSA latent KV offload index logic.

These cover the NPU-independent core: the in-memory offload backend and the A1
gather planning (prefill/decode split + compact remapping). The on-NPU kernel
wiring is verified separately on hardware.
"""

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_backend import (
    InMemoryLatentOffloadBackend,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_manager import (
    INVALID_TOKEN_INDEX,
    SparseOffloadConfig,
    build_gather_plan,
    resolve_scratch_gather,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.runner_integration import (
    build_manager,
    compute_reserved_bytes,
)


LAYER_NAMES = ["L0", "L1"]


def _cpu_config(**overrides):
    kwargs = dict(
        num_layers=2,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        block_size=4,
        max_num_seqs=2,
        topk_tokens=4,
        max_resident_decode_tokens=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return SparseOffloadConfig(**kwargs)


def _build(cfg):
    return build_manager(cfg, LAYER_NAMES)


def test_in_memory_backend_save_load_roundtrip():
    backend = InMemoryLatentOffloadBackend(device="cpu")
    latent_dim = 8
    positions = torch.tensor([0, 1, 2, 3, 4])
    latent = torch.arange(5 * latent_dim, dtype=torch.float32).reshape(5, latent_dim)
    backend.save_layer("L0", "r0", positions, latent)

    # batched load of two requests sharing the same stored data.
    backend.save_layer("L0", "r1", positions, latent + 100)
    load_buffer = torch.zeros((10, latent_dim))
    backend.register_load_buffer(load_buffer)
    backend.set_load_req_ids(["r0", "r1"])
    # r0 wants positions [4, 0], r1 wants [2]; CSR starts [0, 2].
    backend.wait_for_layer_load("L0", [4, 0, 2], [0, 2])
    assert torch.equal(load_buffer[0], latent[4])
    assert torch.equal(load_buffer[1], latent[0])
    assert torch.equal(load_buffer[2], latent[2] + 100)

    backend.free_request("r0")
    assert ("r0", "L0") not in backend._store


def test_gather_plan_splits_prefill_and_decode():
    # prompt_len=10: positions <10 are prefill (LMCache), >=10 are decode (store).
    # request 0 selects [2, 11, 5];  request 1 selects [12, 0, INVALID].
    topk = torch.tensor([[2, 11, 5], [12, 0, INVALID_TOKEN_INDEX]])
    prompt_lens = torch.tensor([10, 10])
    block_size = 4
    blocks_per_req = 1  # topk=3 fits in one block of size 4

    plan = build_gather_plan(topk, prompt_lens, block_size, blocks_per_req)

    # valid counts
    assert plan.seq_lens_kv.tolist() == [3, 2]

    # request 0: prefill positions 2 and 5 at compact slots 0 and 2; decode pos 11->1.
    assert plan.prefill_positions[0].tolist() == [2, INVALID_TOKEN_INDEX, 5]
    assert plan.decode_positions[0].tolist() == [INVALID_TOKEN_INDEX, 1, INVALID_TOKEN_INDEX]

    # request 1: decode pos 12->2 at slot 0; prefill pos 0 at slot 1; last padded.
    assert plan.decode_positions[1].tolist() == [2, INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]
    assert plan.prefill_positions[1].tolist() == [INVALID_TOKEN_INDEX, 0, INVALID_TOKEN_INDEX]

    # compact local indices [0..k-1], padded with INVALID for the empty slot.
    assert plan.sparse_indices[0].tolist() == [0, 1, 2]
    assert plan.sparse_indices[1].tolist() == [0, 1, INVALID_TOKEN_INDEX]

    # dest slots = region_base (b * blocks_per_req * block_size) + local slot.
    assert plan.dest_slot[0].tolist() == [0, 1, 2]
    assert plan.dest_slot[1].tolist() == [4, 5, INVALID_TOKEN_INDEX]

    # each request maps to its own contiguous scratch block.
    assert plan.scratch_block_table.tolist() == [[0], [1]]


def test_gather_plan_multi_block_region():
    # topk=6 with block_size=4 -> 2 blocks per request region.
    topk = torch.tensor([[0, 1, 2, 3, 4, 5]])
    prompt_lens = torch.tensor([100])
    plan = build_gather_plan(topk, prompt_lens, block_size=4, scratch_blocks_per_req=2)
    # all prefill, compact slots 0..5, dest slots 0..5 (region base 0).
    assert plan.sparse_indices[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert plan.dest_slot[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert plan.scratch_block_table.tolist() == [[0, 1]]
    assert plan.seq_lens_kv.tolist() == [6]


def test_gather_plan_all_invalid_row():
    topk = torch.tensor([[INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]])
    prompt_lens = torch.tensor([5])
    plan = build_gather_plan(topk, prompt_lens, block_size=4, scratch_blocks_per_req=1)
    assert plan.seq_lens_kv.tolist() == [0]
    assert plan.dest_slot[0].tolist() == [INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]
    assert plan.sparse_indices[0].tolist() == [INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]


def test_compute_reserved_bytes():
    cfg = _cpu_config()
    # scratch: scratch_blocks_per_req = ceil(4/4)=1, *max_num_seqs(2)=2 blocks
    #          2 blocks * block_size(4) * latent_dim(6) * 4 bytes = 192
    # decode:  num_layers(2)*max_num_seqs(2)*D(8)*latent_dim(6)*4 = 768
    # load:    max_num_seqs(2)*topk(4)*latent_dim(6)*4 = 192
    assert compute_reserved_bytes(cfg) == 192 + 768 + 192


def test_manager_gather_mixed_sources_roundtrip():
    cfg = _cpu_config()
    mgr = _build(cfg)  # in-memory backend on cpu
    latent_dim = cfg.latent_dim

    # Prefill: store prompt tokens 0..4 for layer 1, distinct per-token latent.
    prompt_len = 5
    positions = torch.arange(prompt_len)
    k_nope = torch.arange(prompt_len * cfg.kv_lora_rank, dtype=torch.float32).reshape(
        prompt_len, cfg.kv_lora_rank
    )
    k_pe = torch.arange(prompt_len * cfg.qk_rope_head_dim, dtype=torch.float32).reshape(
        prompt_len, cfg.qk_rope_head_dim
    )
    mgr.store_prefill_layer("r0", "L1", positions, k_nope, k_pe)

    # One decode token already generated -> sequence pos 5 (decode-store row 0).
    dnope = torch.full((1, cfg.kv_lora_rank), 99.0)
    dpe = torch.full((1, cfg.qk_rope_head_dim), 88.0)
    mgr.append_decode_token("r0", "L1", dnope, dpe)

    # Indexer selects prefill pos 3, decode pos 5, prefill pos 1, then padding.
    topk = torch.tensor([[3, 5, 1, INVALID_TOKEN_INDEX]])
    plan = build_gather_plan(
        topk, torch.tensor([prompt_len]), cfg.block_size, cfg.scratch_blocks_per_req
    )
    s_knope, s_kpe, sparse_indices, block_table = mgr.gather_decode_layer("L1", ["r0"], plan)

    knope_flat = s_knope.view(-1, cfg.kv_lora_rank)
    kpe_flat = s_kpe.view(-1, cfg.qk_rope_head_dim)
    # compact slot 0 = prefill pos 3
    assert torch.equal(knope_flat[0], k_nope[3])
    assert torch.equal(kpe_flat[0], k_pe[3])
    # compact slot 1 = decode pos 5 (store row 0)
    assert torch.equal(knope_flat[1], dnope[0])
    assert torch.equal(kpe_flat[1], dpe[0])
    # compact slot 2 = prefill pos 1
    assert torch.equal(knope_flat[2], k_nope[1])
    # kernel args
    assert sparse_indices[0].tolist() == [0, 1, 2, INVALID_TOKEN_INDEX]
    assert block_table.tolist() == [[0]]


def _toy_attention(q, k_full, v_full):
    """Single-query scaled-dot-product attention over a [k, d] KV set."""
    scores = (q @ k_full.t()) / (q.shape[-1] ** 0.5)
    return torch.softmax(scores, dim=-1) @ v_full


def test_offload_path_attends_to_exactly_the_selected_tokens():
    """End-to-end pre-NPU correctness: gathering selected latent into the A1 scratch
    and resolving it the way the kernel will yields the *same* attended K/V (and the
    same attention output) as directly indexing the full latent at the topk
    positions. Decoupled from both NPU and LMCache.
    """
    torch.manual_seed(0)
    cfg = _cpu_config(topk_tokens=6, block_size=4, max_num_seqs=1)
    mgr = _build(cfg)

    prompt_len = 12
    positions = torch.arange(prompt_len)
    k_nope = torch.randn(prompt_len, cfg.kv_lora_rank)
    k_pe = torch.randn(prompt_len, cfg.qk_rope_head_dim)
    mgr.store_prefill_layer("r0", "L0", positions, k_nope, k_pe)

    # two decode tokens already generated -> seq positions 12, 13 (store rows 0, 1).
    # Each step: append for every layer, then advance the per-request row once.
    dec_nope = torch.randn(2, cfg.kv_lora_rank)
    dec_pe = torch.randn(2, cfg.qk_rope_head_dim)
    for i in range(2):
        mgr.append_decode_token("r0", "L0", dec_nope[i : i + 1], dec_pe[i : i + 1])
        mgr.advance_decode_step("r0")

    # Full resident latent (prefill + decode), the ground-truth source.
    full_nope = torch.cat([k_nope, dec_nope], dim=0)
    full_pe = torch.cat([k_pe, dec_pe], dim=0)

    # indexer picks a mix of prefill and decode positions, out of order.
    topk = torch.tensor([[9, 13, 2, 12, 0, INVALID_TOKEN_INDEX]])
    plan = build_gather_plan(
        topk, torch.tensor([prompt_len]), cfg.block_size, cfg.scratch_blocks_per_req
    )
    s_knope, s_kpe, sparse_indices, block_table = mgr.gather_decode_layer("L0", ["r0"], plan)

    got_nope, got_pe = resolve_scratch_gather(
        s_knope, s_kpe, sparse_indices, block_table, cfg.block_size, plan.seq_lens_kv
    )[0]

    valid_positions = torch.tensor([9, 13, 2, 12, 0])
    exp_nope = full_nope.index_select(0, valid_positions)
    exp_pe = full_pe.index_select(0, valid_positions)

    # gathered KV matches the originally-selected tokens, in order.
    assert torch.allclose(got_nope, exp_nope)
    assert torch.allclose(got_pe, exp_pe)

    # and so does the resulting attention output (combine nope+pe as the KV vector).
    q = torch.randn(1, cfg.latent_dim)
    out_offload = _toy_attention(q, torch.cat([got_nope, got_pe], -1), torch.cat([got_nope, got_pe], -1))
    out_full = _toy_attention(q, torch.cat([exp_nope, exp_pe], -1), torch.cat([exp_nope, exp_pe], -1))
    assert torch.allclose(out_offload, out_full)


def test_manager_decode_store_capacity_guard():
    cfg = _cpu_config(max_resident_decode_tokens=1)
    mgr = _build(cfg)
    nope = torch.zeros((1, cfg.kv_lora_rank))
    pe = torch.zeros((1, cfg.qk_rope_head_dim))
    mgr.append_decode_token("r0", "L0", nope, pe)
    mgr.advance_decode_step("r0")
    try:
        mgr.append_decode_token("r0", "L0", nope, pe)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_manager_free_request_recycles_slot():
    cfg = _cpu_config(max_num_seqs=1)
    mgr = _build(cfg)
    nope = torch.zeros((1, cfg.kv_lora_rank))
    pe = torch.zeros((1, cfg.qk_rope_head_dim))
    mgr.append_decode_token("r0", "L0", nope, pe)
    mgr.free_request("r0")
    # slot recycled -> a new request can take it without exhausting max_num_seqs.
    mgr.append_decode_token("r1", "L0", nope, pe)


def test_hooks_store_prefill_splits_requests_by_csr():
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import store_prefill

    cfg = _cpu_config()
    mgr = _build(cfg)
    # two requests packed: r0 has 3 tokens, r1 has 2 tokens.
    qsl = torch.tensor([0, 3, 5])
    ctx = torch.tensor([0, 10])  # r1 already had 10 computed tokens (chunked prefill)
    k_nope = torch.randn(5, cfg.kv_lora_rank)
    k_pe = torch.randn(5, cfg.qk_rope_head_dim)
    store_prefill(mgr, "L0", ["r0", "r1"], qsl, ctx, k_nope, k_pe)

    backend = mgr.backend
    # r0 stored at positions 0..2, r1 at absolute positions 10..11.
    assert ("r0", "L0") in backend._store
    r1 = backend._store[("r1", "L0")]
    assert r1.shape[0] == 12  # dense up to max stored position (11) + 1
    assert torch.equal(r1[10], torch.cat([k_nope[3], k_pe[3]], -1))
    assert torch.equal(r1[11], torch.cat([k_nope[4], k_pe[4]], -1))


def test_hooks_gather_decode_full_step_two_layers():
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import (
        gather_decode,
        store_prefill,
    )

    cfg = _cpu_config(topk_tokens=4, block_size=4, max_num_seqs=1)
    mgr = _build(cfg)
    prompt_len = 6
    # store prompt latent for both layers.
    for ln in LAYER_NAMES:
        kn = torch.randn(prompt_len, cfg.kv_lora_rank)
        kp = torch.randn(prompt_len, cfg.qk_rope_head_dim)
        store_prefill(mgr, ln, ["r0"], torch.tensor([0, prompt_len]), torch.tensor([0]), kn, kp)

    # one decode step over both layers: indexer picks prefill pos 2 + the new token.
    topk = torch.tensor([[2, prompt_len, INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]])
    for ln in LAYER_NAMES:
        cur_nope = torch.randn(1, cfg.kv_lora_rank)
        cur_pe = torch.randn(1, cfg.qk_rope_head_dim)
        sk, skp, si, bt = gather_decode(
            mgr, ln, ["r0"], topk, torch.tensor([prompt_len]), cfg.block_size, cur_nope, cur_pe
        )
        # the new token (decode-store row 0) landed in compact slot 1.
        assert torch.equal(sk.view(-1, cfg.kv_lora_rank)[1], cur_nope[0])
        assert si[0].tolist()[:2] == [0, 1]
    mgr.advance_decode_step("r0")  # once per step, after the final layer
    assert mgr._decode_len["r0"] == 1


def test_hooks_gather_decode_accepts_3d_topk():
    # The Ascend indexer emits [num_tokens, 1, topk]; gather_decode must squeeze it.
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import (
        gather_decode,
        store_prefill,
    )

    cfg = _cpu_config(topk_tokens=4, block_size=4, max_num_seqs=1)
    mgr = _build(cfg)
    prompt_len = 5
    kn = torch.randn(prompt_len, cfg.kv_lora_rank)
    kp = torch.randn(prompt_len, cfg.qk_rope_head_dim)
    store_prefill(mgr, "L0", ["r0"], torch.tensor([0, prompt_len]), torch.tensor([0]), kn, kp)

    topk_3d = torch.tensor([[[2, 0, INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]]])  # [1,1,4]
    cur_nope = torch.randn(1, cfg.kv_lora_rank)
    cur_pe = torch.randn(1, cfg.qk_rope_head_dim)
    sk, skp, si, bt = gather_decode(
        mgr, "L0", ["r0"], topk_3d, torch.tensor([prompt_len]), cfg.block_size, cur_nope, cur_pe
    )
    # slot 0 = prefill pos 2, slot 1 = prefill pos 0.
    assert torch.equal(sk.view(-1, cfg.kv_lora_rank)[0], kn[2])
    assert torch.equal(sk.view(-1, cfg.kv_lora_rank)[1], kn[0])
