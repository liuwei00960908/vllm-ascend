# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstraction boundary for the (layerwise) LMCache latent store.

This is the *only* place that depends on the LMCache layerwise API.

Confirmed LMCache load contract (batched per layer, across the decode batch):

    wait_for_layer_load(layer_name: str,
                        selected_tokens: list[int],
                        token_start_index: list[int]) -> None

  * ``layer_name``   : the attention layer to load.
  * ``selected_tokens``: flat list of token positions to gather for the whole batch.
  * ``token_start_index``: per-request start offsets into ``selected_tokens`` (CSR),
    so LMCache knows which flat entries belong to which request.
  * returns nothing: the loaded latent is written into a buffer registered ahead of
    time via :meth:`register_load_buffer` (connector-style, like ``register_kv_caches``).

ASSUMPTION (INTERFACE-VERIFY with the LMCache author): the loaded latent is written
into the registered load buffer **tightly, in ``selected_tokens`` order** — i.e. flat
row ``j`` of the buffer corresponds to ``selected_tokens[j]``. The manager then copies
those rows into the kernel's block-aligned scratch, so this layout assumption is the
only thing that must hold; everything downstream is decoupled from it.

The MLA latent is two tensors on the NPU (``k_nope`` of ``kv_lora_rank`` and ``k_pe``
of ``qk_rope_head_dim``); the manager packs them into one ``latent`` of width
``kv_lora_rank + qk_rope_head_dim`` for transport, and the load buffer holds that
packed layout.

OPEN (pending the LMCache author): the exact **store/save** signature used at prefill
(``save_layer`` below is a placeholder), and whether load is async (it is named
``wait_for_layer_load``, implying it blocks until ready).
"""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LatentOffloadBackend(Protocol):
    """Layerwise, token-indexed latent store (LMCache adapter contract)."""

    def register_load_buffer(self, load_buffer: torch.Tensor) -> None:
        """Register the buffer ``wait_for_layer_load`` writes loaded latent into.

        Shape ``[max_load_tokens, latent_dim]``; row ``j`` receives the latent for
        ``selected_tokens[j]`` (see module-level ASSUMPTION).
        """
        ...

    def save_layer(
        self,
        layer_name: str,
        req_id: str,
        token_positions: torch.Tensor,
        latent: torch.Tensor,
    ) -> None:
        """Store one layer's prompt latent. Called once per (request, layer) at
        prefill end. NOTE: placeholder signature — align with the LMCache save API.
        """
        ...

    def wait_for_layer_load(
        self,
        layer_name: str,
        selected_tokens: list[int],
        token_start_index: list[int],
    ) -> None:
        """Gather ``selected_tokens`` for ``layer_name`` into the registered load
        buffer (tight, in ``selected_tokens`` order). Blocks until loaded.
        """
        ...

    def free_request(self, req_id: str) -> None:
        """Drop all stored latent for a finished/aborted request."""
        ...


class InMemoryLatentOffloadBackend:
    """Reference backend that keeps latent in memory and faithfully implements the
    confirmed ``wait_for_layer_load`` contract. Lets the full pipeline run before the
    real LMCache adapter exists. NOT memory-relieving (holds latent in a dict).
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self._device = torch.device(device)
        # (req_id, layer_name) -> dense latent indexed by sequence position.
        self._store: dict[tuple[str, str], torch.Tensor] = {}
        self._load_buffer: torch.Tensor | None = None

    def register_load_buffer(self, load_buffer: torch.Tensor) -> None:
        self._load_buffer = load_buffer

    def save_layer(
        self,
        layer_name: str,
        req_id: str,
        token_positions: torch.Tensor,
        latent: torch.Tensor,
    ) -> None:
        positions = token_positions.to(torch.long).cpu()
        max_pos = int(positions.max().item()) + 1 if positions.numel() > 0 else 0
        dense = torch.zeros(
            (max_pos, latent.shape[-1]), dtype=latent.dtype, device=self._device
        )
        dense.index_copy_(0, positions.to(self._device), latent.to(self._device))
        self._store[(req_id, layer_name)] = dense

    def wait_for_layer_load(
        self,
        layer_name: str,
        selected_tokens: list[int],
        token_start_index: list[int],
    ) -> None:
        assert self._load_buffer is not None, "register_load_buffer first"
        # token_start_index gives per-request boundaries; per-request lookup keys are
        # supplied out of band in this reference impl via _load_req_ids (set by the
        # manager just before the call) since the dict is keyed by req_id.
        req_ids = self._load_req_ids
        starts = list(token_start_index) + [len(selected_tokens)]
        for r, req_id in enumerate(req_ids):
            lo, hi = starts[r], starts[r + 1]
            if hi <= lo:
                continue
            dense = self._store[(req_id, layer_name)]
            idx = torch.tensor(selected_tokens[lo:hi], dtype=torch.long, device=self._device)
            self._load_buffer[lo:hi] = dense.index_select(0, idx).to(self._load_buffer.dtype)

    # Reference-only side channel: the real LMCache keys lookups internally; the
    # in-memory impl needs the batch's req_ids to resolve its per-req_id dict.
    _load_req_ids: list[str] = []

    def set_load_req_ids(self, req_ids: list[str]) -> None:
        self._load_req_ids = req_ids

    def free_request(self, req_id: str) -> None:
        for key in [k for k in self._store if k[0] == req_id]:
            del self._store[key]
