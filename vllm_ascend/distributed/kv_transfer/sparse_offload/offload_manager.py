# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side manager for DSA latent KV offload (see DESIGN.md).

Responsibilities:
  * own the two NPU buffers reserved from the KV-cache budget:
      - ``scratch``        : one-layer, reused within a forward (A1 read buffer),
      - ``decode_store``   : persistent, all layers, holds decode-generated latent;
  * pack/unpack the MLA latent (k_nope + k_pe) for the layerwise LMCache backend;
  * at prefill end, push every prompt token's latent to the backend (once);
  * at each decode step/layer, split the indexer top-k into prefill (LMCache) and
    decode (resident store) sources, gather both compactly into ``scratch``, and
    build the kernel arguments (compact ``sparse_indices`` + scratch ``block_table``)
    that point ``npu_sparse_flash_attention`` at the scratch instead of a full
    paged latent cache.

The index-planning core (:func:`build_gather_plan`) is pure tensor logic and is unit
tested on CPU; the device writes around it are thin.

NOTE: the precise semantics of the indexer ``topk_indices`` (sentinel for empty
slots, whether the current token is included) and of ``npu_sparse_flash_attention``'s
``sparse_indices`` / ``block_table`` are validated on NPU hardware in task #5. Points
that depend on on-hardware behavior are marked ``HW-VERIFY``.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_backend import (
    LatentOffloadBackend,
)

# Sentinel marking an unused / padded entry in a top-k row produced by the indexer.
# HW-VERIFY: confirm the value the Ascend indexer emits for padded slots.
INVALID_TOKEN_INDEX = -1


@dataclass
class SparseOffloadConfig:
    num_layers: int
    kv_lora_rank: int  # head_size of k_nope (kv_cache[0])
    qk_rope_head_dim: int  # head_size of k_pe (kv_cache[1])
    block_size: int
    max_num_seqs: int  # caps concurrent running requests -> scratch / store slots
    topk_tokens: int  # indexer index_topk
    max_resident_decode_tokens: int  # D: per-request decode-latent store capacity
    dtype: torch.dtype
    device: torch.device

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
    """

    # [b, topk] gather sources, row-aligned with the *compacted* output order:
    #   prefill entries hold a sequence position (latent in LMCache),
    #   decode entries hold a sequence position (latent in decode_store),
    #   padded entries hold INVALID_TOKEN_INDEX.
    prefill_positions: torch.Tensor  # prefill sources, -1 elsewhere
    decode_positions: torch.Tensor  # decode sources, -1 elsewhere
    # destination scratch slot for each selected token, row-aligned with the two
    # arrays above (the compaction): [b, topk], -1 for padded entries.
    dest_slot: torch.Tensor
    # kernel args pointing at the scratch:
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
        topk_indices: int tensor ``[b, topk]`` of selected sequence positions per
            decode query (one query per request at decode), ``INVALID_TOKEN_INDEX``
            for unused slots.
        prompt_lens: int tensor ``[b]`` prompt length per request; positions
            ``< prompt_len`` are prefill (LMCache), ``>= prompt_len`` are decode
            (resident store, store-row = pos - prompt_len).
        block_size: paged block size of the scratch.
        scratch_blocks_per_req: contiguous scratch blocks reserved per request.

    Returns:
        GatherPlan with sources and the remapped kernel arguments.
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

    prefill_positions = torch.where(
        is_prefill, topk_indices, torch.full_like(topk_indices, INVALID_TOKEN_INDEX)
    )
    decode_positions = torch.where(
        is_decode,
        topk_indices - prompt_lens.unsqueeze(1),
        torch.full_like(topk_indices, INVALID_TOKEN_INDEX),
    )

    # Destination scratch slot = request's block region base + local_slot.
    region_base = (
        torch.arange(b, dtype=torch.long).unsqueeze(1)
        * scratch_blocks_per_req
        * block_size
    )
    dest_slot = torch.where(
        valid,
        region_base + local_slot,
        torch.full_like(topk_indices, INVALID_TOKEN_INDEX),
    )

    # Compact local indices fed to the kernel (resolved against scratch_block_table).
    sparse_indices = torch.where(
        valid, local_slot, torch.full_like(topk_indices, INVALID_TOKEN_INDEX)
    )

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
    (``layout_kv="PA_BSND"``, ``sparse_block_size=1``). This reconstructs, per
    request, the ``(k_nope, k_pe)`` rows the kernel would attend to, in attend order.

    Used by tests to prove the A1 gather + remap feeds the kernel exactly the
    originally-selected token latents. HW-VERIFY: this encodes our assumption about
    the kernel's index/block_table semantics; confirmed against the real kernel on
    NPU in task #5.
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
    """Owns the scratch / decode-store buffers and drives store + gather."""

    def __init__(
        self,
        config: SparseOffloadConfig,
        backend: LatentOffloadBackend,
        layer_names: list[str],
        scratch_knope: torch.Tensor,
        scratch_kpe: torch.Tensor,
        decode_store: torch.Tensor,
        load_buffer: torch.Tensor,
    ) -> None:
        """Buffers are allocated by the model runner (from the reserved KV budget)
        and handed in here so this manager never allocates device memory itself.

        Args:
            layer_names: ordered SFA attention layer names; the position in this list
                is the contiguous ``layer_id`` used to index ``decode_store``.
            scratch_knope: [scratch_num_blocks, block_size, 1, kv_lora_rank]
            scratch_kpe:   [scratch_num_blocks, block_size, 1, qk_rope_head_dim]
            decode_store:  [num_layers, max_num_seqs, D, latent_dim]
            load_buffer:   [max_num_seqs * topk_tokens, latent_dim] — LMCache writes
                loaded prefill latent here (registered with the backend).
        """
        self.config = config
        self.backend = backend
        self._layer_id = {name: i for i, name in enumerate(layer_names)}
        self._scratch_knope = scratch_knope
        self._scratch_kpe = scratch_kpe
        self._decode_store = decode_store
        self._load_buffer = load_buffer
        backend.register_load_buffer(load_buffer)

        # request slot bookkeeping for the decode store (small free-list).
        self._free_slots: list[int] = list(range(config.max_num_seqs))
        self._req_slot: dict[str, int] = {}

    # ------------------------------------------------------------------ slots
    def _slot_for(self, req_id: str) -> int:
        slot = self._req_slot.get(req_id)
        if slot is None:
            if not self._free_slots:
                raise RuntimeError(
                    "SparseLatentOffloadManager: no free decode-store slot; "
                    "running requests exceeded max_num_seqs."
                )
            slot = self._free_slots.pop()
            self._req_slot[req_id] = slot
        return slot

    def free_request(self, req_id: str) -> None:
        slot = self._req_slot.pop(req_id, None)
        if slot is not None:
            self._free_slots.append(slot)
        self.backend.free_request(req_id)

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
    def append_decode_token(
        self,
        req_id: str,
        layer_name: str,
        decode_row: int,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        """Write the current decode token's latent into the resident store at an
        explicit row (= current absolute position - prompt_len).

        Must be called before :meth:`gather_decode_layer` for the same step so the
        token is visible if the indexer selects it. Using an explicit row (rather than
        an internal counter) means every layer writes the same row and there is no
        per-step "advance" to time — the row is derived from the token position.
        """
        if decode_row >= self.config.max_resident_decode_tokens:
            raise RuntimeError(
                f"SparseLatentOffloadManager: request {req_id} exceeded "
                f"max_resident_decode_tokens={self.config.max_resident_decode_tokens}; "
                "v1 keeps decode latent resident on NPU."
            )
        slot = self._slot_for(req_id)
        latent = self._pack(k_nope, k_pe)
        self._decode_store[self._layer_id[layer_name], slot, decode_row] = latent.squeeze(0)

    def gather_decode_layer(
        self,
        layer_name: str,
        req_ids: list[str],
        plan: GatherPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize the selected latent for one layer into the scratch.

        Loads all requests' selected prefill tokens for this layer with a single
        batched ``wait_for_layer_load`` (into the registered load buffer), then copies
        the loaded prefill latent + resident decode latent into the block-aligned A1
        scratch. Returns ``(scratch_knope, scratch_kpe, sparse_indices,
        scratch_block_table)`` ready to feed ``npu_sparse_flash_attention``.
        """
        cfg = self.config
        layer_id = self._layer_id[layer_name]

        # --- 1. Build the batched LMCache load request (flat prefill positions). ---
        # plan tensors are on CPU (build_gather_plan moved them), so .tolist() here is
        # cheap (no device sync). The one unavoidable sync is moving topk_indices to
        # CPU once per layer inside build_gather_plan.
        flat_selected: list[int] = []
        token_start_index: list[int] = []
        # cache per-request masks/slots for the copy phase below.
        per_req = []
        for b, req_id in enumerate(req_ids):
            dest = plan.dest_slot[b]
            valid = dest != INVALID_TOKEN_INDEX
            pref_pos = plan.prefill_positions[b]
            dec_pos = plan.decode_positions[b]
            is_pref = valid & (pref_pos != INVALID_TOKEN_INDEX)
            is_dec = valid & (dec_pos != INVALID_TOKEN_INDEX)
            token_start_index.append(len(flat_selected))
            pref_positions = pref_pos[is_pref].tolist()
            flat_selected.extend(pref_positions)
            per_req.append((req_id, dest, is_pref, is_dec, pref_pos, dec_pos))

        if hasattr(self.backend, "set_load_req_ids"):  # in-memory reference backend
            self.backend.set_load_req_ids(req_ids)
        self.backend.wait_for_layer_load(layer_name, flat_selected, token_start_index)

        # --- 2. Copy loaded prefill + resident decode latent into the scratch. ---
        knope_flat = self._scratch_knope.view(-1, cfg.kv_lora_rank)
        kpe_flat = self._scratch_kpe.view(-1, cfg.qk_rope_head_dim)
        dev = self._scratch_knope.device
        for b, (req_id, dest, is_pref, is_dec, pref_pos, dec_pos) in enumerate(per_req):
            lo = token_start_index[b]
            n_pref = int(is_pref.sum())
            if n_pref:
                pref_latent = self._load_buffer[lo : lo + n_pref]
                knope, kpe = self._unpack(pref_latent.to(cfg.dtype))
                dst = dest[is_pref].to(dev)
                knope_flat.index_copy_(0, dst, knope.to(dev))
                kpe_flat.index_copy_(0, dst, kpe.to(dev))
            if bool(is_dec.any()):
                slot = self._req_slot[req_id]
                rows = dec_pos[is_dec].to(self._decode_store.device)
                dec_latent = self._decode_store[layer_id, slot].index_select(0, rows)
                knope, kpe = self._unpack(dec_latent.to(cfg.dtype))
                dst = dest[is_dec].to(dev)
                knope_flat.index_copy_(0, dst, knope.to(dev))
                kpe_flat.index_copy_(0, dst, kpe.to(dev))

        # Match the native kernel arg dtypes (int32) to avoid a dtype mismatch.
        return (
            self._scratch_knope,
            self._scratch_kpe,
            plan.sparse_indices.to(dev, torch.int32),
            plan.scratch_block_table.to(dev, torch.int32),
            plan.seq_lens_kv.to(dev, torch.int32),  # per-req valid count -> actual_seq_lengths_kv
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
