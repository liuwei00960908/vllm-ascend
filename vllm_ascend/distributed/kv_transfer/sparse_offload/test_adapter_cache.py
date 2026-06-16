# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity test for :mod:`adapter_cache` — no NPU, no vLLM required.

Run:  python test_adapter_cache.py

It proves the translation chain end-to-end on CPU:
  position -> logical block id -> adapter.load -> physical slot -> block_table,
and that the kernel, resolving ``block_table[p//bs]*bs + p%bs`` over the pool,
reads exactly the latent that was stored/written for the selected tokens.

The values (layers, cap, ratio, block_size, dims, topk) are chosen small here ONLY
for the test; the module itself hard-codes none of them.
"""

from __future__ import annotations

import os
import sys

import torch

# Make `kv_cache_adapter` and this module importable WITHOUT installing the package
# (its setup.py compiles NPU/CUDA .so; the CPU path here needs none of that).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _ensure_kv_cache_adapter_importable() -> None:
    try:
        import kv_cache_adapter  # noqa: F401
        return
    except ImportError:
        pass
    # Override with env var, else assume kv_cache_adapter sits next to the
    # vllm-ascend repo root (…/<root>/kv_cache_adapter and …/<root>/vllm-ascend).
    parent = os.environ.get("KV_CACHE_ADAPTER_PARENT")
    if parent is None:
        parent = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
    if parent not in sys.path:
        sys.path.insert(0, parent)


_ensure_kv_cache_adapter_importable()

from adapter_cache import AdapterCacheConfig, AdapterLatentCache  # noqa: E402
from kv_cache_adapter import InMemoryBlockStoreBackend  # noqa: E402

torch.manual_seed(0)
DEVICE = torch.device("cpu")
DTYPE = torch.float32  # fp32 on CPU; bf16 layout identical, this just eases allclose


def make_config():
    return AdapterCacheConfig(
        layer_names=["l0", "l1"],
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        block_size=4,
        topk=6,
        max_model_len=64,
        max_num_seqs=4,
        pool_concurrency_cap=2,
        # 4.0 here only so the small test's spread-out topk (7 distinct blocks)
        # fits; production ratio must be derived from real topk block-clustering.
        pool_ratio=4.0,
        dtype=DTYPE,
        device=DEVICE,
    )


def ref_knope(req_slot: int, pos: int, dim: int) -> torch.Tensor:
    base = float(req_slot * 10_000 + pos * 10)
    return base + torch.arange(dim, dtype=DTYPE) * 0.01


def ref_kpe(req_slot: int, pos: int, dim: int) -> torch.Tensor:
    base = float(req_slot * 10_000 + pos * 10)
    return base + 0.5 + torch.arange(dim, dtype=DTYPE) * 0.01


def build_with_shared_backends(cfg):
    """Cache whose per-layer InMemory backends we keep handles to (to seed prefill)."""
    backends: dict[str, InMemoryBlockStoreBackend] = {}

    def factory(layer_name: str) -> InMemoryBlockStoreBackend:
        b = InMemoryBlockStoreBackend(num_logical_blocks=cfg.num_logical_blocks)
        backends[layer_name] = b
        return b

    return AdapterLatentCache(cfg, backend_factory=factory), backends


def seed_prefill(cfg, backends, layer_name, req_slot, num_prefill_tokens):
    """Store full blocks of synthetic prefill latent into the layer backend."""
    bs, kdim, pdim = cfg.block_size, cfg.kv_lora_rank, cfg.qk_rope_head_dim
    n_blocks = (num_prefill_tokens + bs - 1) // bs
    ids, knope_blocks, kpe_blocks = [], [], []
    for blk in range(n_blocks):
        kn = torch.zeros(bs, 1, kdim, dtype=DTYPE)
        kp = torch.zeros(bs, 1, pdim, dtype=DTYPE)
        for off in range(bs):
            pos = blk * bs + off
            if pos >= num_prefill_tokens:
                break
            kn[off, 0] = ref_knope(req_slot, pos, kdim)
            kp[off, 0] = ref_kpe(req_slot, pos, pdim)
        ids.append(blk + req_slot * cfg.blocks_per_req)
        knope_blocks.append(kn)
        kpe_blocks.append(kp)
    backends[layer_name].save_blocks(
        torch.tensor(ids, dtype=torch.int64),
        [torch.stack(knope_blocks), torch.stack(kpe_blocks)],
    )


def resolve_and_check(cfg, res, req_slots, expected_knope, expected_kpe, label):
    """Mirror the kernel's read and compare to the expected latent per selected token."""
    bs = cfg.block_size
    knope_flat = res.knope_pool.reshape(-1, cfg.kv_lora_rank)
    kpe_flat = res.kpe_pool.reshape(-1, cfg.qk_rope_head_dim)
    b, topk = res.sparse_indices.shape
    checked = 0
    for r in range(b):
        for j in range(topk):
            p = int(res.sparse_indices[r, j])
            if p < 0:
                continue
            slot = int(res.block_table[r, p // bs])
            phys = slot * bs + (p % bs)
            got_kn = knope_flat[phys]
            got_kp = kpe_flat[phys]
            exp_kn = expected_knope(int(req_slots[r]), p)
            exp_kp = expected_kpe(int(req_slots[r]), p)
            assert torch.allclose(got_kn, exp_kn), f"[{label}] knope mismatch r={r} p={p}\n got {got_kn}\n exp {exp_kn}"
            assert torch.allclose(got_kp, exp_kp), f"[{label}] kpe mismatch r={r} p={p}"
            checked += 1
    assert checked > 0, f"[{label}] nothing checked"
    return checked


def test_retrieve_prefill():
    cfg = make_config()
    cache, backends = build_with_shared_backends(cfg)
    rsA = cache.req_slot("A")
    rsB = cache.req_slot("B")
    for ln in cfg.layer_names:
        seed_prefill(cfg, backends, ln, rsA, num_prefill_tokens=20)
        seed_prefill(cfg, backends, ln, rsB, num_prefill_tokens=12)

    req_slots = torch.tensor([rsA, rsB], dtype=torch.int64)
    # topk includes duplicates within a block (dedup path) and -1 padding.
    topk = torch.tensor(
        [[0, 1, 5, 9, 17, -1],     # A: positions across blocks 0,1,2,4 (1&0 same block)
         [2, 3, 7, 11, -1, -1]],   # B: blocks 0,1,2
        dtype=torch.int64,
    )
    res = cache.retrieve("l0", req_slots, topk)
    n = resolve_and_check(cfg, res, req_slots,
                          lambda rs, p: ref_knope(rs, p, cfg.kv_lora_rank),
                          lambda rs, p: ref_kpe(rs, p, cfg.qk_rope_head_dim), "prefill")
    cache.release_after_fa("l0", res.loaded_ids)
    print(f"  test_retrieve_prefill: {n} selected tokens verified")


def test_insert_then_retrieve():
    cfg = make_config()
    cache, backends = build_with_shared_backends(cfg)
    rsA = cache.req_slot("A")
    for ln in cfg.layer_names:
        seed_prefill(cfg, backends, ln, rsA, num_prefill_tokens=8)

    # Generate decode tokens at positions 8,9,10 (a fresh block 2) for layer l0.
    written = {}
    for pos in (8, 9, 10):
        kn = ref_knope(rsA, pos, cfg.kv_lora_rank)
        kp = ref_kpe(rsA, pos, cfg.qk_rope_head_dim)
        cache.insert_decode_token("l0", "A", pos, kn, kp)
        written[pos] = (kn, kp)

    # topk mixing prefill (0,3) and freshly-inserted decode tokens (8,9,10).
    req_slots = torch.tensor([rsA], dtype=torch.int64)
    topk = torch.tensor([[0, 3, 8, 9, 10, -1]], dtype=torch.int64)
    res = cache.retrieve("l0", req_slots, topk)
    n = resolve_and_check(cfg, res, req_slots,
                          lambda rs, p: ref_knope(rs, p, cfg.kv_lora_rank),
                          lambda rs, p: ref_kpe(rs, p, cfg.qk_rope_head_dim), "insert")
    cache.release_after_fa("l0", res.loaded_ids)
    print(f"  test_insert_then_retrieve: {n} selected tokens verified (prefill+decode mixed)")


def test_hit_skips_refetch():
    cfg = make_config()
    cache, backends = build_with_shared_backends(cfg)
    rsA = cache.req_slot("A")
    seed_prefill(cfg, backends, "l0", rsA, num_prefill_tokens=8)
    req_slots = torch.tensor([rsA], dtype=torch.int64)
    topk = torch.tensor([[0, 1, 4, 5, -1, -1]], dtype=torch.int64)

    res1 = cache.retrieve("l0", req_slots, topk)
    loads_after_first = len(backends["l0"].load_calls)
    cache.release_after_fa("l0", res1.loaded_ids)

    res2 = cache.retrieve("l0", req_slots, topk)  # same blocks -> all hits
    loads_after_second = len(backends["l0"].load_calls)
    cache.release_after_fa("l0", res2.loaded_ids)

    assert loads_after_second == loads_after_first, (
        f"second retrieve re-fetched (hits should skip backend): "
        f"{loads_after_first} -> {loads_after_second}"
    )
    print(f"  test_hit_skips_refetch: second retrieve fetched 0 new blocks "
          f"(backend load count stayed {loads_after_first})")


def test_reserved_bytes_and_sizes():
    cfg = make_config()
    cache, _ = build_with_shared_backends(cfg)
    # num_actual = ceil(4.0 * 2 * 6 / 4) = 12
    assert cfg.num_actual_blocks == 12, cfg.num_actual_blocks
    # num_logical = max_num_seqs * ceil(64/4) = 4 * 16 = 64
    assert cfg.num_logical_blocks == 64, cfg.num_logical_blocks
    elt = torch.empty((), dtype=DTYPE).element_size()
    expect = 12 * 4 * (8 + 4) * elt * len(cfg.layer_names)
    assert cache.reserved_bytes() == expect, (cache.reserved_bytes(), expect)
    print(f"  test_reserved_bytes_and_sizes: num_actual=12 num_logical=64 "
          f"reserved={cache.reserved_bytes()} bytes")


if __name__ == "__main__":
    print("CPU parity tests for adapter_cache:")
    test_reserved_bytes_and_sizes()
    test_retrieve_prefill()
    test_insert_then_retrieve()
    test_hit_skips_refetch()
    print("ALL PASSED")
