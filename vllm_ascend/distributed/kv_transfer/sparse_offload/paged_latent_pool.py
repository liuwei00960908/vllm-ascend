# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-managed paged latent store for Route 1 (free prefill latent from NPU).

The fused op ``npu_kv_rmsnorm_rope_cache`` writes the MLA latent into a single
per-layer ``(k_cache, ckv_cache)`` tensor at ``index`` (a slot mapping); it has no
no-cache mode and only writes the latent (not the indexer key, which is a separate
scatter). So to keep the latent OUT of vLLM's full-context paged cache we give the op
*our own*, smaller latent tensors + slot mapping, and read them back in attention via
*our own* block table — exactly like the paged cache, but:

  * sized for concurrent-prefill + decode latent (not full-context * all requests),
  * with our own block allocator so a request's **prefill** latent blocks are freed
    right after prefill (the content is offloaded to LMCache), recycling them.

This is the storage foundation (R1a). Wiring the op write + attention read + the SFA
KVCacheSpec shrink to it is R1b/R1c (NPU-coupled). The block bookkeeping is pure
Python and unit-tested on CPU; only the latent tensors live on device.
"""

import torch


class PagedLatentPool:
    """Per-layer paged latent (``knope``/``kpe``) with our own block allocator.

    Layout mirrors the vLLM paged latent so the op/kernel are happy:
        knope: [num_layers, num_blocks, block_size, 1, kv_lora_rank]
        kpe:   [num_layers, num_blocks, block_size, 1, qk_rope_head_dim]
    The block table is shared across layers (a token occupies the same block/offset in
    every layer); each layer has its own slice of storage.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.knope = torch.zeros(
            (num_layers, num_blocks, block_size, 1, kv_lora_rank), dtype=dtype, device=self.device
        )
        self.kpe = torch.zeros(
            (num_layers, num_blocks, block_size, 1, qk_rope_head_dim), dtype=dtype, device=self.device
        )
        self._free: list[int] = list(range(num_blocks))
        self._req_blocks: dict[str, list[int]] = {}

    # ----------------------------------------------------------- per-layer view
    def layer_caches(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(knope, kpe)`` for one layer — pass as the op's ckv_cache/k_cache and as
        the attention's key tensors."""
        return self.knope[layer_id], self.kpe[layer_id]

    # ----------------------------------------------------------- allocation
    def reserve(self, req_id: str, num_positions: int) -> None:
        """Ensure the request owns enough blocks to cover ``num_positions`` tokens."""
        blocks = self._req_blocks.setdefault(req_id, [])
        need = (num_positions + self.block_size - 1) // self.block_size
        while len(blocks) < need:
            if not self._free:
                raise RuntimeError(
                    "PagedLatentPool exhausted; increase the pool size or free finished "
                    "requests (prefill blocks should be freed after prefill)."
                )
            blocks.append(self._free.pop())

    def free_request(self, req_id: str) -> None:
        self._free.extend(self._req_blocks.pop(req_id, []))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    # ----------------------------------------------------------- addressing
    def slot_mapping(self, req_id: str, positions: torch.Tensor) -> torch.Tensor:
        """Map sequence ``positions`` -> flat pool slots (block*block_size + offset),
        for the op's ``index`` argument. ``reserve`` must have covered these positions.
        """
        blocks = torch.tensor(self._req_blocks[req_id], dtype=torch.long, device=positions.device)
        pos = positions.to(torch.long)
        return blocks[pos // self.block_size] * self.block_size + pos % self.block_size

    def gather(
        self, req_id: str, layer_id: int, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read ``(knope, kpe)`` rows for a layer at the given sequence positions."""
        slots = self.slot_mapping(req_id, positions).to(self.device)
        knope = self.knope[layer_id].reshape(-1, self.knope.shape[-1]).index_select(0, slots)
        kpe = self.kpe[layer_id].reshape(-1, self.kpe.shape[-1]).index_select(0, slots)
        return knope, kpe

    def store(
        self,
        req_id: str,
        layer_id: int,
        positions: torch.Tensor,
        knope_rows: torch.Tensor,
        kpe_rows: torch.Tensor,
    ) -> None:
        """Scatter ``(knope_rows, kpe_rows)`` for a layer at the given positions
        (reserving blocks as needed). Used to populate the pool from exec_kv's return."""
        self.reserve(req_id, int(positions.max().item()) + 1 if positions.numel() else 0)
        slots = self.slot_mapping(req_id, positions).to(self.device)
        self.knope[layer_id].reshape(-1, self.knope.shape[-1]).index_copy_(
            0, slots, knope_rows.to(self.device, self.knope.dtype)
        )
        self.kpe[layer_id].reshape(-1, self.kpe.shape[-1]).index_copy_(
            0, slots, kpe_rows.to(self.device, self.kpe.dtype)
        )

    def block_table(self, req_id: str, width: int) -> torch.Tensor:
        """Padded ``[width]`` block-id row for the attention kernel."""
        blocks = self._req_blocks[req_id]
        row = torch.zeros(width, dtype=torch.int32, device=self.device)
        if blocks:
            row[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=self.device)
        return row

    def allocated_bytes(self) -> int:
        return (
            self.knope.numel() * self.knope.element_size()
            + self.kpe.numel() * self.kpe.element_size()
        )
