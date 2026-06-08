# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the DSA latent KV offload index logic.

Covers the NPU-independent core: the in-memory offload backend, the A1 gather
planning (prefill/decode split + compact remapping), and the manager gather that
reads prefill latent from the backend (LMCache) and decode latent from a (simulated)
paged latent cache. The on-NPU kernel wiring is verified separately by the parity run.
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
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return SparseOffloadConfig(**kwargs)


def _build(cfg):
    return build_manager(cfg, LAYER_NAMES)


def _make_paged(cfg, num_blocks=4):
    """A simulated paged latent cache (kv_cache[0], kv_cache[1]) + a 1-request
    block_table that maps logical block i -> physical block i."""
    kv0 = torch.zeros((num_blocks, cfg.block_size, 1, cfg.kv_lora_rank))
    kv1 = torch.zeros((num_blocks, cfg.block_size, 1, cfg.qk_rope_head_dim))
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)  # [1, num_blocks]
    return kv0, kv1, block_table


def _write_paged(kv0, kv1, block_table, b, pos, nope, pe, block_size):
    slot = int(block_table[b][pos // block_size]) * block_size + pos % block_size
    kv0.view(-1, kv0.shape[-1])[slot] = nope
    kv1.view(-1, kv1.shape[-1])[slot] = pe


def test_in_memory_backend_save_load_roundtrip():
    backend = InMemoryLatentOffloadBackend(device="cpu")
    latent_dim = 8
    positions = torch.tensor([0, 1, 2, 3, 4])
    latent = torch.arange(5 * latent_dim, dtype=torch.float32).reshape(5, latent_dim)
    backend.save_layer("L0", "r0", positions, latent)

    backend.save_layer("L0", "r1", positions, latent + 100)
    load_buffer = torch.zeros((10, latent_dim))
    backend.register_load_buffer(load_buffer)
    backend.set_load_req_ids(["r0", "r1"])
    backend.wait_for_layer_load("L0", [4, 0, 2], [0, 2])  # r0:[4,0], r1:[2]
    assert torch.equal(load_buffer[0], latent[4])
    assert torch.equal(load_buffer[1], latent[0])
    assert torch.equal(load_buffer[2], latent[2] + 100)

    backend.free_request("r0")
    assert ("r0", "L0") not in backend._store


def test_gather_plan_splits_prefill_and_decode():
    # prompt_len=10: positions <10 prefill (LMCache), >=10 decode (paged). Both arrays
    # hold ABSOLUTE positions; only the source differs.
    topk = torch.tensor([[2, 11, 5], [12, 0, INVALID_TOKEN_INDEX]])
    prompt_lens = torch.tensor([10, 10])
    plan = build_gather_plan(topk, prompt_lens, block_size=4, scratch_blocks_per_req=1)

    assert plan.seq_lens_kv.tolist() == [3, 2]
    assert plan.prefill_positions[0].tolist() == [2, INVALID_TOKEN_INDEX, 5]
    assert plan.decode_positions[0].tolist() == [INVALID_TOKEN_INDEX, 11, INVALID_TOKEN_INDEX]
    assert plan.decode_positions[1].tolist() == [12, INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]
    assert plan.prefill_positions[1].tolist() == [INVALID_TOKEN_INDEX, 0, INVALID_TOKEN_INDEX]
    assert plan.sparse_indices[0].tolist() == [0, 1, 2]
    assert plan.sparse_indices[1].tolist() == [0, 1, INVALID_TOKEN_INDEX]
    assert plan.dest_slot[0].tolist() == [0, 1, 2]
    assert plan.dest_slot[1].tolist() == [4, 5, INVALID_TOKEN_INDEX]
    assert plan.scratch_block_table.tolist() == [[0], [1]]


def test_gather_plan_multi_block_region():
    topk = torch.tensor([[0, 1, 2, 3, 4, 5]])
    prompt_lens = torch.tensor([100])
    plan = build_gather_plan(topk, prompt_lens, block_size=4, scratch_blocks_per_req=2)
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


def test_compute_reserved_bytes():
    cfg = _cpu_config()
    # scratch: blocks_per_req=ceil(4/4)=1, *max_num_seqs(2)=2 blocks; 2*4*6*4 = 192
    # load:    max_num_seqs(2)*topk(4)*latent_dim(6)*4 = 192    (no decode store)
    assert compute_reserved_bytes(cfg) == 192 + 192


def test_manager_gather_mixed_sources_roundtrip():
    cfg = _cpu_config()
    mgr = _build(cfg)

    prompt_len = 5
    positions = torch.arange(prompt_len)
    k_nope = torch.arange(prompt_len * cfg.kv_lora_rank, dtype=torch.float32).reshape(
        prompt_len, cfg.kv_lora_rank
    )
    k_pe = torch.arange(prompt_len * cfg.qk_rope_head_dim, dtype=torch.float32).reshape(
        prompt_len, cfg.qk_rope_head_dim
    )
    mgr.store_prefill_layer("r0", "L1", positions, k_nope, k_pe)

    # one decode token already generated at abs pos 5, resident in the paged cache.
    kv0, kv1, block_table = _make_paged(cfg)
    dnope = torch.full((cfg.kv_lora_rank,), 99.0)
    dpe = torch.full((cfg.qk_rope_head_dim,), 88.0)
    _write_paged(kv0, kv1, block_table, 0, 5, dnope, dpe, cfg.block_size)

    # indexer selects prefill 3, decode 5, prefill 1, then padding.
    topk = torch.tensor([[3, 5, 1, INVALID_TOKEN_INDEX]])
    plan = build_gather_plan(topk, torch.tensor([prompt_len]), cfg.block_size, cfg.scratch_blocks_per_req)
    s_knope, s_kpe, sparse_indices, block_tbl, seq_lens_kv = mgr.gather_decode_layer(
        "L1", ["r0"], plan, (kv0, kv1), block_table, cfg.block_size
    )

    knope_flat = s_knope.view(-1, cfg.kv_lora_rank)
    kpe_flat = s_kpe.view(-1, cfg.qk_rope_head_dim)
    assert torch.equal(knope_flat[0], k_nope[3])   # prefill 3 (backend)
    assert torch.equal(kpe_flat[0], k_pe[3])
    assert torch.equal(knope_flat[1], dnope)       # decode 5 (paged)
    assert torch.equal(kpe_flat[1], dpe)
    assert torch.equal(knope_flat[2], k_nope[1])   # prefill 1 (backend)
    assert sparse_indices[0].tolist() == [0, 1, 2, INVALID_TOKEN_INDEX]
    assert block_tbl.tolist() == [[0]]


def _toy_attention(q, k_full, v_full):
    scores = (q @ k_full.t()) / (q.shape[-1] ** 0.5)
    return torch.softmax(scores, dim=-1) @ v_full


def test_offload_path_attends_to_exactly_the_selected_tokens():
    """Pre-NPU correctness: gather selected latent into the A1 scratch (prefill from
    backend, decode from paged) and resolve it the way the kernel will -> identical
    attended K/V and attention output vs directly indexing the full latent."""
    torch.manual_seed(0)
    cfg = _cpu_config(topk_tokens=6, block_size=4, max_num_seqs=1)
    mgr = _build(cfg)

    prompt_len = 12
    k_nope = torch.randn(prompt_len, cfg.kv_lora_rank)
    k_pe = torch.randn(prompt_len, cfg.qk_rope_head_dim)
    mgr.store_prefill_layer("r0", "L0", torch.arange(prompt_len), k_nope, k_pe)

    # two decode tokens at abs pos 12, 13, resident in the paged cache.
    kv0, kv1, block_table = _make_paged(cfg, num_blocks=6)
    dec_nope = torch.randn(2, cfg.kv_lora_rank)
    dec_pe = torch.randn(2, cfg.qk_rope_head_dim)
    for i in range(2):
        _write_paged(kv0, kv1, block_table, 0, 12 + i, dec_nope[i], dec_pe[i], cfg.block_size)

    full_nope = torch.cat([k_nope, dec_nope], dim=0)
    full_pe = torch.cat([k_pe, dec_pe], dim=0)

    topk = torch.tensor([[9, 13, 2, 12, 0, INVALID_TOKEN_INDEX]])
    plan = build_gather_plan(topk, torch.tensor([prompt_len]), cfg.block_size, cfg.scratch_blocks_per_req)
    s_knope, s_kpe, sparse_indices, block_tbl, seq_lens_kv = mgr.gather_decode_layer(
        "L0", ["r0"], plan, (kv0, kv1), block_table, cfg.block_size
    )

    got_nope, got_pe = resolve_scratch_gather(
        s_knope, s_kpe, sparse_indices, block_tbl, cfg.block_size, seq_lens_kv
    )[0]
    valid_positions = torch.tensor([9, 13, 2, 12, 0])
    exp_nope = full_nope.index_select(0, valid_positions)
    exp_pe = full_pe.index_select(0, valid_positions)
    assert torch.allclose(got_nope, exp_nope)
    assert torch.allclose(got_pe, exp_pe)

    q = torch.randn(1, cfg.latent_dim)
    out_offload = _toy_attention(q, torch.cat([got_nope, got_pe], -1), torch.cat([got_nope, got_pe], -1))
    out_full = _toy_attention(q, torch.cat([exp_nope, exp_pe], -1), torch.cat([exp_nope, exp_pe], -1))
    assert torch.allclose(out_offload, out_full)


def test_manager_free_request_delegates_to_backend():
    cfg = _cpu_config()
    mgr = _build(cfg)
    mgr.store_prefill_layer(
        "r0", "L0", torch.arange(3),
        torch.randn(3, cfg.kv_lora_rank), torch.randn(3, cfg.qk_rope_head_dim),
    )
    mgr.free_request("r0")
    assert ("r0", "L0") not in mgr.backend._store


def test_hooks_store_prefill_splits_requests_by_csr():
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import store_prefill

    cfg = _cpu_config()
    mgr = _build(cfg)
    qsl = torch.tensor([0, 3, 5])
    ctx = torch.tensor([0, 10])  # r1 had 10 prior tokens (chunked prefill)
    k_nope = torch.randn(5, cfg.kv_lora_rank)
    k_pe = torch.randn(5, cfg.qk_rope_head_dim)
    store_prefill(mgr, "L0", ["r0", "r1"], qsl, ctx, k_nope, k_pe)

    r1 = mgr.backend._store[("r1", "L0")]
    assert r1.shape[0] == 12  # dense up to abs pos 11
    assert torch.equal(r1[10], torch.cat([k_nope[3], k_pe[3]], -1))
    assert torch.equal(r1[11], torch.cat([k_nope[4], k_pe[4]], -1))


def test_hooks_gather_decode_full_step():
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import (
        gather_decode,
        store_prefill,
    )

    cfg = _cpu_config(topk_tokens=4, block_size=4, max_num_seqs=1)
    mgr = _build(cfg)
    prompt_len = 6
    kn = torch.randn(prompt_len, cfg.kv_lora_rank)
    kp = torch.randn(prompt_len, cfg.qk_rope_head_dim)
    store_prefill(mgr, "L0", ["r0"], torch.tensor([0, prompt_len]), torch.tensor([0]), kn, kp)

    # current decode token at abs pos 6, resident in paged cache.
    kv0, kv1, block_table = _make_paged(cfg, num_blocks=4)
    cur_nope = torch.randn(cfg.kv_lora_rank)
    cur_pe = torch.randn(cfg.qk_rope_head_dim)
    _write_paged(kv0, kv1, block_table, 0, 6, cur_nope, cur_pe, cfg.block_size)

    # indexer picks prefill pos 2 + the new token (abs pos 6). 3-D shape [1,1,topk].
    topk = torch.tensor([[[2, 6, INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]]])
    sk, skp, si, bt, sl = gather_decode(
        mgr, "L0", ["r0"], topk, torch.tensor([prompt_len]),
        cfg.block_size, (kv0, kv1), block_table,
    )
    assert torch.equal(sk.view(-1, cfg.kv_lora_rank)[0], kn[2])    # prefill 2
    assert torch.equal(sk.view(-1, cfg.kv_lora_rank)[1], cur_nope)  # decode 6 (paged)
    assert si[0].tolist()[:2] == [0, 1]
    assert sl[0] == 2
