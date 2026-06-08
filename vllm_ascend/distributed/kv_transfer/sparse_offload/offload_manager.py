# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side manager for DSA latent KV offload (see DESIGN.md).

Responsibilities:
  * own the ``scratch`` pool (one-layer, reused within a forward — the A1 read buffer)
    and the ``load_buffer`` that LMCache writes loaded prefill latent into;
  * pack/unpack the MLA latent (k_nope + k_pe) for the layerwise LMCache backend;
  * at prefill end, push every prompt token's latent to the backend (once);
  * at each decode step/layer, split the indexer top-k into prefill (LMCache) and
    decode sources, gather both compactly into ``scratch``, and build the kernel args
    (compact ``sparse_indices`` + scratch ``block_table``) that point
    ``npu_sparse_flash_attention`` at the scratch instead of the full paged latent.

Decode-generated tokens are NOT kept in a separate fixed-size store: their latent is
already resident in the paged latent cache (``exec_kv`` writes it every step, vLLM
manages its growth up to ``max_model_len``). So decode-selected tokens are read back
directly from the paged cache via the request's ``block_table`` — no per-request
length cap. The current step's just-generated token is likewise in the paged cache,
so it needs no special handling.

The index-planning core (:func:`build_gather_plan`) is pure tensor logic, unit tested
on CPU. ``INVALID_TOKEN_INDEX`` (-1) is the indexer's padding sentinel (confirmed on
NPU); the kernel index/block_table semantics are validated by the parity run.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_backend import (
    LatentOffloadBackend,
)

# Sentinel marking an unused / padded entry in a top-k row produced by the indexer.
INVALID_TOKEN_INDEX = -1


@dataclass
class SparseOffloadConfig:
    num_layers: int
    kv_lora_rank: int  # head_size of k_nope (kv_cache[0])
    qk_rope_head_dim: int  # head_size of k_pe (kv_cache[1])
    block_size: int
    max_num_seqs: int  # caps concurrent running requests -> scratch capacity
    topk_tokens: int  # indexer index_topk (== single-layer resident size K)
    dtype: torch.dtype
    device: torch.device
    pool_num_blocks: int = 0  # PagedLatentPool size (Route 1); 0 -> derived default

    @property
    def latent_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def scratch_blocks_per_req(self) -> int:
        return (self.topk_tokens + self.block_size - 1) // self.block_size

    @property
    def scratch_num_blocks(self) -> int:
        return self.scratch_blocks_per_req * self.max_num_seqs


@dataclass
class GatherPlan:
    """CPU-side plan for one decode batch (output of :func:`build_gather_plan`).

    All tensors are int64 on CPU. ``b`` indexes the decode requests in batch order.
    ``prefill_positions`` holds *absolute* positions (LMCache keys); ``decode_positions``
    holds positions *relative to the prompt* (indices into the decode-latent pool).
    """

    prefill_positions: torch.Tensor  # [b, topk] prefill sources (abs pos), -1 elsewhere
    decode_positions: torch.Tensor  # [b, topk] decode sources (rel pos), -1 elsewhere
    dest_slot: torch.Tensor  # [b, topk] compact scratch slot per selected token, -1 pad
    sparse_indices: torch.Tensor  # [b, topk] compact local indices [0..k_b-1], padded
    scratch_block_table: torch.Tensor  # [b, scratch_blocks_per_req] scratch block ids
    seq_lens_kv: torch.Tensor  # [b] number of valid selected tokens k_b


def build_gather_plan(
    topk_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    block_size: int,
    scratch_blocks_per_req: int,
) -> GatherPlan:
    """Plan the A1 compact gather for a decode batch. Pure / CPU-testable.

    Args:
        topk_indices: int tensor ``[b, topk]`` of selected *absolute* sequence
            positions per decode query, ``INVALID_TOKEN_INDEX`` for unused slots.
        prompt_lens: int tensor ``[b]`` prompt length per request; positions
            ``< prompt_len`` are prefill (LMCache), ``>= prompt_len`` are decode
            (read from the paged latent cache).
        block_size: paged block size of the scratch.
        scratch_blocks_per_req: contiguous scratch blocks reserved per request.
    """
    topk_indices = topk_indices.to(torch.long).cpu()
    prompt_lens = prompt_lens.to(torch.long).cpu()
    b, topk = topk_indices.shape

    valid = topk_indices != INVALID_TOKEN_INDEX
    # Compact each row: valid entries get local slots 0,1,2,... in selection order.
    local_slot = torch.cumsum(valid.to(torch.long), dim=1) - 1  # [b, topk]
    seq_lens_kv = valid.sum(dim=1)  # [b], k_b

    is_decode = valid & (topk_indices >= prompt_lens.unsqueeze(1))
    is_prefill = valid & ~is_decode

    invalid = torch.full_like(topk_indices, INVALID_TOKEN_INDEX)
    # Both hold ABSOLUTE positions; source differs: prefill -> LMCache (key=abs pos),
    # decode -> PagedLatentPool (read at abs pos via the request's pool block table).
    prefill_positions = torch.where(is_prefill, topk_indices, invalid)
    decode_positions = torch.where(is_decode, topk_indices, invalid)

    # Destination scratch slot = request's block region base + local_slot.
    region_base = (
        torch.arange(b, dtype=torch.long).unsqueeze(1)
        * scratch_blocks_per_req
        * block_size
    )
    dest_slot = torch.where(valid, region_base + local_slot, invalid)

    # Compact local indices fed to the kernel (resolved against scratch_block_table).
    sparse_indices = torch.where(valid, local_slot, invalid)

    block_ids = torch.arange(b * scratch_blocks_per_req, dtype=torch.long)
    scratch_block_table = block_ids.view(b, scratch_blocks_per_req)

    return GatherPlan(
        prefill_positions=prefill_positions,
        decode_positions=decode_positions,
        dest_slot=dest_slot,
        sparse_indices=sparse_indices,
        scratch_block_table=scratch_block_table,
        seq_lens_kv=seq_lens_kv,
    )


def resolve_scratch_gather(
    scratch_knope: torch.Tensor,
    scratch_kpe: torch.Tensor,
    sparse_indices: torch.Tensor,
    scratch_block_table: torch.Tensor,
    block_size: int,
    seq_lens_kv: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Mirror of how ``npu_sparse_flash_attention`` resolves its KV reads.

    For each request and each valid compact index ``i`` in ``sparse_indices``, the
    physical scratch slot is ``block_table[i // block_size] * block_size + i % block_size``
    (``layout_kv="PA_BSND"``, ``sparse_block_size=1``). Reconstructs, per request, the
    ``(k_nope, k_pe)`` rows the kernel attends to, in order. Used by tests to prove the
    A1 gather + remap feeds the kernel exactly the originally-selected token latents.
    """
    knope_flat = scratch_knope.reshape(-1, scratch_knope.shape[-1])
    kpe_flat = scratch_kpe.reshape(-1, scratch_kpe.shape[-1])
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for b in range(sparse_indices.shape[0]):
        k = int(seq_lens_kv[b])
        local = sparse_indices[b, :k].to(torch.long)
        phys = (
            scratch_block_table[b][local // block_size] * block_size
            + local % block_size
        )
        out.append((knope_flat.index_select(0, phys), kpe_flat.index_select(0, phys)))
    return out


class SparseLatentOffloadManager:
    """Owns the scratch / load buffers and drives prefill store + decode gather."""

    def __init__(
        self,
        config: SparseOffloadConfig,
        backend: LatentOffloadBackend,
        layer_names: list[str],
        scratch_knope: torch.Tensor,
        scratch_kpe: torch.Tensor,
        load_buffer: torch.Tensor,
        decode_pool,
        paged_latent_pool=None,
    ) -> None:
        """Buffers are allocated by the model runner (from the reserved KV budget) and
        handed in here so this manager never allocates device memory itself.

        Shapes:
            scratch_knope: [scratch_num_blocks, block_size, 1, kv_lora_rank]
            scratch_kpe:   [scratch_num_blocks, block_size, 1, qk_rope_head_dim]
            load_buffer:   [max_num_seqs * topk_tokens, latent_dim] — LMCache writes
                loaded prefill latent here (registered with the backend).
            decode_pool:   GrowingDecodeLatentPool — on-demand store for decode latent.
        """
        self.config = config
        self.backend = backend
        self._layer_id = {name: i for i, name in enumerate(layer_names)}
        self._scratch_knope = scratch_knope
        self._scratch_kpe = scratch_kpe
        self._load_buffer = load_buffer
        self._decode_pool = decode_pool
        self._paged_latent_pool = paged_latent_pool
        backend.register_load_buffer(load_buffer)

    def free_request(self, req_id: str) -> None:
        self.backend.free_request(req_id)
        self._decode_pool.free_request(req_id)
        if self._paged_latent_pool is not None:
            self._paged_latent_pool.free_request(req_id)

    # ----------------------------------------- Route 1: paged latent pool (R1b)
    def populate_pool_layer(
        self,
        req_ids: list[str],
        layer_name: str,
        query_start_loc: torch.Tensor,
        context_lens: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        """Scatter this prefill chunk's latent for one layer into the PagedLatentPool
        (positionally), reserving pool blocks per request. Latent comes from exec_kv's
        return; vLLM's paged latent is still written by the op (kept for parity)."""
        pool = self._paged_latent_pool
        layer_id = self._layer_id[layer_name]
        kn = k_nope.reshape(-1, self.config.kv_lora_rank)
        kp = k_pe.reshape(-1, self.config.qk_rope_head_dim)
        qsl = query_start_loc.to(torch.long).tolist()
        ctx = context_lens.to(torch.long).tolist()
        for b, req_id in enumerate(req_ids):
            lo, hi = qsl[b], qsl[b + 1]
            if hi <= lo:
                continue
            positions = torch.arange(ctx[b], ctx[b] + (hi - lo))
            pool.store(req_id, layer_id, positions, kn[lo:hi], kp[lo:hi])

    def pool_exec_kv_slots(
        self,
        layer_name: str,
        req_ids: list[str],
        query_start_loc: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reserve pool blocks for this step's tokens and return ``(pool_slots,
        knope, kpe)`` so exec_kv's op writes latent into the pool (FREE_PAGED mode).
        ``pool_slots`` is aligned with the packed token order (= the vLLM slot_mapping
        order); knope/kpe are this layer's pool tensors (the op's ckv_cache/k_cache)."""
        pool = self._paged_latent_pool
        qsl = query_start_loc.to(torch.long).tolist()
        ctx = context_lens.to(torch.long).tolist()
        parts = []
        for b, req_id in enumerate(req_ids):
            lo, hi = qsl[b], qsl[b + 1]
            if hi <= lo:
                continue
            positions = torch.arange(ctx[b], ctx[b] + (hi - lo))
            pool.reserve(req_id, ctx[b] + (hi - lo))
            parts.append(pool.slot_mapping(req_id, positions))
        pool_slots = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
        knope, kpe = pool.layer_caches(self._layer_id[layer_name])
        return pool_slots.to(knope.device), knope, kpe

    def pool_attn_args(
        self, layer_name: str, req_ids: list[str], max_blocks: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(knope, kpe, block_table)`` for prefill attention over the pool.
        ``block_table`` is ``[num_reqs, max_blocks]`` of pool block ids."""
        pool = self._paged_latent_pool
        knope, kpe = pool.layer_caches(self._layer_id[layer_name])
        bt = torch.stack([pool.block_table(r, max_blocks) for r in req_ids])
        return knope, kpe, bt

    # ----------------------------------------------------------------- prefill
    def store_prefill_layer(
        self,
        req_id: str,
        layer_name: str,
        token_positions: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        """Push one layer's prompt latent to the backend (called once per layer)."""
        latent = self._pack(k_nope, k_pe)
        self.backend.save_layer(layer_name, req_id, token_positions, latent)

    # ------------------------------------------------------------------ decode
    def store_decode_token(
        self,
        req_id: str,
        layer_name: str,
        position: int,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        """Write this step's generated token latent into the PagedLatentPool at its
        ABSOLUTE position. Called once per layer per step (before the gather)."""
        self._paged_latent_pool.store(
            req_id,
            self._layer_id[layer_name],
            torch.tensor([position], dtype=torch.long),
            k_nope.reshape(1, -1),
            k_pe.reshape(1, -1),
        )

    # ------------------------------------------------------------------ decode
    def gather_decode_layer(
        self,
        layer_name: str,
        req_ids: list[str],
        plan: GatherPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize the selected latent for one layer into the scratch.

        Prefill-selected tokens are loaded from the backend (LMCache) via a single
        batched ``wait_for_layer_load`` into the registered load buffer; decode-selected
        tokens are read from the growing decode-latent pool. Both are copied into the
        block-aligned A1 scratch.

        Returns ``(scratch_knope, scratch_kpe, sparse_indices, scratch_block_table,
        seq_lens_kv)`` for ``npu_sparse_flash_attention``.
        """
        cfg = self.config
        dev = self._scratch_knope.device
        layer_id = self._layer_id[layer_name]

        # --- 1. Build the batched LMCache load request (flat prefill positions). ---
        # plan tensors are on CPU, so .tolist() is cheap (no extra device sync).
        flat_selected: list[int] = []
        token_start_index: list[int] = []
        per_req = []
        for b, req_id in enumerate(req_ids):
            dest = plan.dest_slot[b]
            valid = dest != INVALID_TOKEN_INDEX
            pref_pos = plan.prefill_positions[b]
            dec_pos = plan.decode_positions[b]
            is_pref = valid & (pref_pos != INVALID_TOKEN_INDEX)
            is_dec = valid & (dec_pos != INVALID_TOKEN_INDEX)
            token_start_index.append(len(flat_selected))
            flat_selected.extend(pref_pos[is_pref].tolist())
            per_req.append((b, req_id, dest, is_pref, is_dec, dec_pos))

        if hasattr(self.backend, "set_load_req_ids"):  # in-memory reference backend
            self.backend.set_load_req_ids(req_ids)
        self.backend.wait_for_layer_load(layer_name, flat_selected, token_start_index)

        # --- 2. Copy loaded prefill (backend) + decode (pool) latent into scratch. ---
        knope_flat = self._scratch_knope.view(-1, cfg.kv_lora_rank)
        kpe_flat = self._scratch_kpe.view(-1, cfg.qk_rope_head_dim)
        for b, req_id, dest, is_pref, is_dec, dec_pos in per_req:
            lo = token_start_index[b]
            n_pref = int(is_pref.sum())
            if n_pref:
                pref_latent = self._load_buffer[lo : lo + n_pref]
                knope, kpe = self._unpack(pref_latent.to(cfg.dtype))
                dst = dest[is_pref].to(dev)
                knope_flat.index_copy_(0, dst, knope.to(dev))
                kpe_flat.index_copy_(0, dst, kpe.to(dev))
            if bool(is_dec.any()):
                # decode-selected tokens live in the PagedLatentPool at their absolute
                # positions; read them via the request's pool block table.
                knope, kpe = self._paged_latent_pool.gather(
                    req_id, layer_id, dec_pos[is_dec]
                )
                dst = dest[is_dec].to(dev)
                knope_flat.index_copy_(0, dst, knope.to(dev, cfg.dtype))
                kpe_flat.index_copy_(0, dst, kpe.to(dev, cfg.dtype))

        # Match the native kernel arg dtypes (int32). sparse_indices stays 2-D here;
        # the caller adds the singleton head dim for the kernel.
        return (
            self._scratch_knope,
            self._scratch_kpe,
            plan.sparse_indices.to(dev, torch.int32),
            plan.scratch_block_table.to(dev, torch.int32),
            plan.seq_lens_kv.to(dev, torch.int32),
        )

    # ------------------------------------------------------------------ packing
    def _pack(self, k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
        """[*, kv_lora_rank], [*, qk_rope_head_dim] -> [*, latent_dim]."""
        k_nope = k_nope.reshape(-1, self.config.kv_lora_rank)
        k_pe = k_pe.reshape(-1, self.config.qk_rope_head_dim)
        return torch.cat([k_nope, k_pe], dim=-1)

    def _unpack(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[*, latent_dim] -> ([*, kv_lora_rank], [*, qk_rope_head_dim])."""
        return latent.split([self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1)
