# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA latent hot-cache backed by the colleague's ``KVCacheAdapter``.

This wires the on-NPU latent pool (the adapter) into the DSA decode path so the
sparse-attention kernel reads the resident pool *in place* (zero-copy), with the
adapter handling residency (hit/miss) and eviction-to-LMCache.

Design (aligned in review, see DESIGN.md / INTEGRATION.md):
  * Two pools per layer (``k_nope`` of ``kv_lora_rank`` and ``k_pe`` of
    ``qk_rope_head_dim``) share **one** slot mapping — block ``i`` lives at the
    same physical slot in both pools. This mirrors how vLLM already allocates the
    MLA latent as two independent contiguous tensors (``kv_cache[0]``/``[1]``).
  * The pool tensor has **no request dimension**. Requests are distinguished by
    the per-request rows of the kernel ``block_table`` (native paged-attention
    mechanism); the adapter keeps requests disjoint because the logical block id
    encodes the request slot.
  * Retrieve: topk absolute positions -> logical block ids -> ``adapter.load`` ->
    physical slots -> scatter into ``block_table``; ``sparse_indices`` stays the
    original absolute positions (native contract, block_table swapped to point at
    the pool). No data copy.
  * Insert (decode token): ``adapter.load(load_missing=False)`` hands back a slot;
    the caller writes the new token's latent straight into ``pool[slot, offset]``.
    The block is resident from allocation, so a later topk hit serves it directly.
  * Eviction (pool full) is handled inside the adapter: it writes the evicted
    block back through its backend (LMCache for the real run).

Everything here is parameter-driven via :class:`AdapterCacheConfig`; nothing about
layer count / concurrency / sizes is hard-coded. The module depends only on
``torch`` and the ``kv_cache_adapter`` package, so the CPU parity test runs without
vLLM or an NPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from kv_cache_adapter import (  # type: ignore[import-not-found]
    BlockStoreBackend,
    InMemoryBlockStoreBackend,
    KVCacheAdapter,
)

ID_DTYPE = torch.int64
INVALID_POSITION = -1


@dataclass
class AdapterCacheConfig:
    """All sizing is derived from these fields; callers pass real values in."""

    layer_names: list[str]
    kv_lora_rank: int          # k_nope width (kv_cache[0])
    qk_rope_head_dim: int      # k_pe width (kv_cache[1])
    block_size: int            # tokens per block; align with LMCache chunk size
    topk: int                  # indexer index_topk (selected tokens per query)
    max_model_len: int         # bounds blocks-per-request (logical id space)
    max_num_seqs: int          # bounds the req-slot id space (recycled on free)
    pool_concurrency_cap: int  # concurrency the pool is sized to hold without thrash
    pool_ratio: float          # headroom over the working set (>=1; more -> more reuse)
    dtype: torch.dtype
    device: torch.device

    @property
    def latent_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def blocks_per_req(self) -> int:
        return math.ceil(self.max_model_len / self.block_size)

    @property
    def num_logical_blocks(self) -> int:
        # req-slot space x blocks-per-request; the id space, not physical memory.
        return self.max_num_seqs * self.blocks_per_req

    @property
    def num_actual_blocks(self) -> int:
        # physical slots per layer = ceil(ratio * cap * topk / block_size).
        tokens = self.pool_ratio * self.pool_concurrency_cap * self.topk
        return max(1, math.ceil(tokens / self.block_size))


@dataclass
class RetrieveResult:
    """Kernel arguments for ``npu_sparse_flash_attention`` (pool read in place)."""

    knope_pool: torch.Tensor          # (num_actual_blocks, block_size, 1, kv_lora_rank)
    kpe_pool: torch.Tensor            # (num_actual_blocks, block_size, 1, qk_rope_head_dim)
    block_table: torch.Tensor         # [b, blocks_per_req] int32, pool slot per req-block
    sparse_indices: torch.Tensor      # [b, topk] int32, absolute positions (-1 padded)
    seq_lens: torch.Tensor            # [b] int32, valid selected tokens per request
    loaded_ids: torch.Tensor          # unique logical ids pinned by this call (for release)


class _LayerState:
    __slots__ = ("knope_pool", "kpe_pool", "adapter", "backend")

    def __init__(self, knope_pool, kpe_pool, adapter, backend):
        self.knope_pool = knope_pool
        self.kpe_pool = kpe_pool
        self.adapter = adapter
        self.backend = backend


class AdapterLatentCache:
    """Per-layer adapter-backed latent hot cache (decision: one adapter per layer)."""

    def __init__(
        self,
        config: AdapterCacheConfig,
        backend_factory: Callable[[str], BlockStoreBackend] | None = None,
    ) -> None:
        self.config = config
        self._layers: dict[str, _LayerState] = {}
        self._req_slot_of: dict[str, int] = {}
        self._free_req_slots: list[int] = list(range(config.max_num_seqs))
        # remembers the slot of the decode block each request is currently filling,
        # so per-token writes don't re-allocate within a block.
        self._decode_block: dict[tuple[str, str], tuple[int, int]] = {}

        for layer_name in config.layer_names:
            knope_pool = torch.zeros(
                (config.num_actual_blocks, config.block_size, 1, config.kv_lora_rank),
                dtype=config.dtype,
                device=config.device,
            )
            kpe_pool = torch.zeros(
                (config.num_actual_blocks, config.block_size, 1, config.qk_rope_head_dim),
                dtype=config.dtype,
                device=config.device,
            )
            backend = (
                backend_factory(layer_name)
                if backend_factory is not None
                else InMemoryBlockStoreBackend(num_logical_blocks=config.num_logical_blocks)
            )
            adapter = KVCacheAdapter(
                config.num_actual_blocks,
                config.num_logical_blocks,
                [knope_pool, kpe_pool],
                backend,
            )
            self._layers[layer_name] = _LayerState(knope_pool, kpe_pool, adapter, backend)

    # ----------------------------------------------------------------- sizing
    def reserved_bytes(self) -> int:
        cfg = self.config
        elt = torch.empty((), dtype=cfg.dtype).element_size()
        per_layer = cfg.num_actual_blocks * cfg.block_size * cfg.latent_dim * elt
        return per_layer * len(cfg.layer_names)

    # ----------------------------------------------------------- req-slot map
    def req_slot(self, req_id: str) -> int:
        slot = self._req_slot_of.get(req_id)
        if slot is None:
            if not self._free_req_slots:
                raise RuntimeError("no free req-slot; exceeded max_num_seqs")
            slot = self._free_req_slots.pop()
            self._req_slot_of[req_id] = slot
        return slot

    def free_request(self, req_id: str) -> None:
        slot = self._req_slot_of.get(req_id)
        if slot is None:
            return
        # Release the still-pinned current decode block of every layer so the pool
        # can reclaim/spill it; then recycle the req-slot id.
        for layer_name, layer in self._layers.items():
            cached = self._decode_block.pop((req_id, layer_name), None)
            if cached is not None:
                logical = torch.tensor(
                    [cached[0] + slot * self.config.blocks_per_req], dtype=ID_DTYPE, device=self.config.device
                )
                layer.adapter.release(logical)
        self._req_slot_of.pop(req_id, None)
        self._free_req_slots.append(slot)

    # ----------------------------------------------------------- logical ids
    def _logical(self, req_slot: int, positions: torch.Tensor) -> torch.Tensor:
        """absolute positions -> global logical block ids (request-scoped)."""
        return positions // self.config.block_size + req_slot * self.config.blocks_per_req

    # -------------------------------------------------------------- retrieve
    def retrieve(
        self,
        layer_name: str,
        req_slots: torch.Tensor,       # [b] int64, req-slot per batch row
        topk_positions: torch.Tensor,  # [b, topk] int64, absolute positions, -1 padded
    ) -> RetrieveResult:
        cfg = self.config
        layer = self._layers[layer_name]
        dev = cfg.device
        bs = cfg.block_size
        b, topk = topk_positions.shape

        valid = topk_positions >= 0
        local_block = torch.where(valid, topk_positions // bs, torch.zeros_like(topk_positions))
        logical = local_block + req_slots[:, None] * cfg.blocks_per_req  # [b, topk]
        logical = torch.where(valid, logical, torch.full_like(logical, -1))

        valid_logical = logical[valid]
        if valid_logical.numel() > 0:
            unique_ids = torch.unique(valid_logical).to(ID_DTYPE)
            slots = layer.adapter.load(unique_ids, load_missing=True)
        else:
            unique_ids = logical.new_zeros((0,), dtype=ID_DTYPE)
            slots = unique_ids.clone()

        # dense logical-id -> slot lookup (only loaded ids are valid).
        slot_of = torch.zeros(cfg.num_logical_blocks, dtype=ID_DTYPE, device=dev)
        if unique_ids.numel() > 0:
            slot_of[unique_ids] = slots.to(ID_DTYPE)

        block_table = torch.zeros(b, cfg.blocks_per_req, dtype=ID_DTYPE, device=dev)
        if valid.any():
            sel_slot = slot_of[logical.clamp(min=0)]            # [b, topk]
            rows = torch.arange(b, device=dev)[:, None].expand(b, topk)
            block_table[rows[valid], local_block[valid]] = sel_slot[valid]

        return RetrieveResult(
            knope_pool=layer.knope_pool,
            kpe_pool=layer.kpe_pool,
            block_table=block_table.to(torch.int32),
            sparse_indices=topk_positions.to(torch.int32),
            seq_lens=valid.sum(dim=1).to(torch.int32),
            loaded_ids=unique_ids,
        )

    def release_after_fa(self, layer_name: str, loaded_ids: torch.Tensor) -> None:
        if loaded_ids.numel() == 0:
            return
        self._layers[layer_name].adapter.release(loaded_ids)

    # ---------------------------------------------------------------- insert
    def insert_decode_token(
        self,
        layer_name: str,
        req_id: str,
        position: int,
        k_nope: torch.Tensor,   # (kv_lora_rank,)
        k_pe: torch.Tensor,     # (qk_rope_head_dim,)
    ) -> int:
        """Write one generated token's latent into the pool; return its slot.

        A new block id is allocated lazily (``load_missing=False`` -> a free slot,
        no backend fetch) when this token starts a fresh block; subsequent tokens
        of the same block reuse the remembered slot.
        """
        cfg = self.config
        layer = self._layers[layer_name]
        req_slot = self.req_slot(req_id)
        block_local = position // cfg.block_size
        offset = position % cfg.block_size
        key = (req_id, layer_name)

        cached = self._decode_block.get(key)
        if cached is None or cached[0] != block_local:
            # Starting a fresh block: unpin the previous (now-complete) block so it
            # becomes evictable, then allocate a slot for the new one (no fetch).
            if cached is not None:
                prev_logical = torch.tensor(
                    [cached[0] + req_slot * cfg.blocks_per_req], dtype=ID_DTYPE, device=cfg.device
                )
                layer.adapter.release(prev_logical)
            logical = torch.tensor(
                [block_local + req_slot * cfg.blocks_per_req], dtype=ID_DTYPE, device=cfg.device
            )
            slot = int(layer.adapter.load(logical, load_missing=False)[0])
            self._decode_block[key] = (block_local, slot)
        else:
            slot = cached[1]

        layer.knope_pool[slot, offset, 0, :] = k_nope.to(cfg.dtype)
        layer.kpe_pool[slot, offset, 0, :] = k_pe.to(cfg.dtype)
        return slot
