# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A separate, on-demand-growing paged store for decode-generated latent (Stage2-B).

DSA offload frees the *prefill* latent from the NPU (it lives in LMCache). The
*decode*-generated tokens stay resident, but their count grows unboundedly with the
generation length, so we cannot pre-reserve a fixed buffer for them. This module is a
small paged cache dedicated to decode latent:

  * memory is allocated **on demand, one chunk of blocks at a time** (not all up
    front) — when a request's decode tokens fill a block, a new block is handed out,
    and a new chunk is allocated only when the free list is empty;
  * it has its own block allocator + per-request block lists, fully independent of the
    indexer-key paged cache and its full-sequence ``slot_mapping``;
  * a decode token at decode-index ``d`` (= absolute_pos - prompt_len) for a request
    lives at ``req_blocks[d // block_size]`` block, offset ``d % block_size``, across
    all layers (the block layout is shared across layers; the tensor has a layer dim).

The block-bookkeeping is pure Python and unit-tested on CPU; only the chunk tensors
live on device.

NOTE: growth allocates device memory outside vLLM's scheduler KV-budget accounting.
That is acceptable because Stage2-B frees the (much larger) prefill latent from the
paged cache; the decode pool grows slowly into that freed headroom. Production should
bound/cap it or integrate with the block budget.
"""

import torch


class GrowingDecodeLatentPool:
    """On-demand paged store for decode latent, packed as ``[kv_lora_rank|qk_rope]``."""

    def __init__(
        self,
        num_layers: int,
        block_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        chunk_blocks: int = 64,
    ) -> None:
        self.num_layers = num_layers
        self.block_size = block_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.latent_dim = kv_lora_rank + qk_rope_head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.chunk_blocks = chunk_blocks

        # chunk c holds global blocks [c*chunk_blocks, (c+1)*chunk_blocks);
        # each chunk tensor is [chunk_blocks, num_layers, block_size, latent_dim].
        self._chunks: list[torch.Tensor] = []
        self._free: list[int] = []  # recycled global block ids
        self._next_block: int = 0  # next never-yet-allocated global block id
        self._req_blocks: dict[str, list[int]] = {}  # req_id -> global block ids (decode order)

    # ------------------------------------------------------------- allocation
    def _grow_one_chunk(self) -> None:
        chunk = torch.zeros(
            (self.chunk_blocks, self.num_layers, self.block_size, self.latent_dim),
            dtype=self.dtype,
            device=self.device,
        )
        base = len(self._chunks) * self.chunk_blocks
        self._chunks.append(chunk)
        self._free.extend(range(base, base + self.chunk_blocks))

    def _alloc_block(self) -> int:
        if not self._free:
            self._grow_one_chunk()  # grow only when out of blocks
        return self._free.pop()

    def _block_view(self, gid: int) -> torch.Tensor:
        """[num_layers, block_size, latent_dim] view of global block ``gid``."""
        return self._chunks[gid // self.chunk_blocks][gid % self.chunk_blocks]

    # ----------------------------------------------------------------- writes
    def append_token(
        self,
        req_id: str,
        layer_id: int,
        decode_index: int,
        latent: torch.Tensor,
    ) -> None:
        """Write one decode token's packed latent (``[latent_dim]``) for a layer.

        ``decode_index`` is the token's position relative to the prompt. Every layer
        writes the same ``decode_index`` (same block/offset); blocks are grown lazily.
        """
        blocks = self._req_blocks.setdefault(req_id, [])
        bi = decode_index // self.block_size
        while len(blocks) <= bi:  # grow this request's block list one block at a time
            blocks.append(self._alloc_block())
        self._block_view(blocks[bi])[layer_id, decode_index % self.block_size] = latent

    # ---------------------------------------------------------------- reads
    def gather(
        self,
        req_id: str,
        layer_id: int,
        decode_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather packed latent ``[n, latent_dim]`` for the given decode indices.

        ``decode_indices`` are positions relative to the prompt (long tensor). Returns
        rows in the same order.
        """
        blocks = self._req_blocks[req_id]
        idx = decode_indices.to(torch.long).tolist()
        out = torch.empty((len(idx), self.latent_dim), dtype=self.dtype, device=self.device)
        bs = self.block_size
        for i, d in enumerate(idx):
            out[i] = self._block_view(blocks[d // bs])[layer_id, d % bs]
        return out

    # ----------------------------------------------------------------- free
    def free_request(self, req_id: str) -> None:
        self._free.extend(self._req_blocks.pop(req_id, []))

    # ----------------------------------------------------------------- stats
    @property
    def num_allocated_blocks(self) -> int:
        return len(self._chunks) * self.chunk_blocks

    def allocated_bytes(self) -> int:
        from vllm.utils.torch_utils import get_dtype_size

        per_block = self.num_layers * self.block_size * self.latent_dim
        return self.num_allocated_blocks * per_block * get_dtype_size(self.dtype)
