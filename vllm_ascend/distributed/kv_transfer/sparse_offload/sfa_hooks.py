# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks called from AscendSFAImpl to drive DSA latent offload (see DESIGN.md).

These compose the tested manager primitives into the two call sites in the SFA
attention forward:

  * :func:`store_prefill` — after the prompt latent is computed, push each request's
    tokens to LMCache (once per layer).
  * :func:`gather_decode` — after the indexer produces ``topk_indices`` and before
    ``npu_sparse_flash_attention``, materialize the selected latent into the A1
    scratch and return the kernel arguments to use instead of the paged latent cache.

The functions take everything explicitly so they are unit-testable off-NPU. The SFA
call site must extract these from its locals / ``attn_metadata``; the extraction
points are marked ``HW-VERIFY`` in the sfa_v1.py edits (field names depend on the
Ascend metadata layout).
"""

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_manager import (
    SparseLatentOffloadManager,
    build_gather_plan,
)


def store_prefill(
    manager: SparseLatentOffloadManager,
    layer_name: str,
    req_ids: list[str],
    query_start_loc: torch.Tensor,
    context_lens: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
) -> None:
    """Offload this layer's freshly-computed prompt latent, per request.

    Args:
        query_start_loc: int tensor ``[num_reqs + 1]`` CSR offsets of each request's
            tokens within the packed ``k_nope``/``k_pe`` (cu_seqlens style).
        context_lens: int tensor ``[num_reqs]`` already-computed tokens per request
            before this step (0 for a single-shot prefill; >0 with chunked prefill),
            so absolute positions are ``context_len + local_offset``.
        k_nope: ``[num_tokens, kv_lora_rank]`` packed across requests.
        k_pe:   ``[num_tokens, qk_rope_head_dim]`` packed across requests.
    """
    qsl = query_start_loc.tolist()
    ctx = context_lens.tolist()
    for b, req_id in enumerate(req_ids):
        lo, hi = qsl[b], qsl[b + 1]
        if hi <= lo:
            continue
        positions = torch.arange(ctx[b], ctx[b] + (hi - lo))
        manager.store_prefill_layer(
            req_id, layer_name, positions, k_nope[lo:hi], k_pe[lo:hi]
        )


def gather_decode(
    manager: SparseLatentOffloadManager,
    layer_name: str,
    req_ids: list[str],
    topk_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    cur_positions: torch.Tensor,
    block_size: int,
    cur_k_nope: torch.Tensor,
    cur_k_pe: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append the current decode token, then gather the selected latent into scratch.

    Args:
        topk_indices: ``[num_decode_reqs, topk]`` (or ``[*, 1, topk]``) indexer output,
            absolute sequence positions, ``-1`` padded.
        prompt_lens:  ``[num_decode_reqs]`` prompt length per request (the prefill /
            decode boundary).
        cur_positions: ``[num_decode_reqs]`` absolute position of this step's new token
            per request (= seq_len - 1); the decode-store row is ``pos - prompt_len``.
        cur_k_nope / cur_k_pe: ``[num_decode_reqs, *]`` the current step's latent for
            this layer (one new token per decode request), kept resident on the NPU.

    Returns ``(scratch_knope, scratch_kpe, sparse_indices, scratch_block_table)`` for
    ``npu_sparse_flash_attention``.

    The Ascend indexer emits ``topk_indices`` as ``[num_tokens, 1, topk]`` (confirmed
    on NPU); collapse the singleton middle dim to ``[num_decode_reqs, topk]``.
    """
    if topk_indices.dim() == 3:
        topk_indices = topk_indices[:, 0, :]

    # The new token must be visible to the gather if the indexer selected it. The row
    # is the token's position relative to the prompt; every layer writes the same row.
    # cur_positions is on NPU (from seq_lens), prompt_lens on CPU (from numpy); compute
    # the decode rows on CPU (the .tolist() below forces a D2H sync anyway).
    rows = (cur_positions.cpu().to(torch.long) - prompt_lens.cpu().to(torch.long)).tolist()
    for b, req_id in enumerate(req_ids):
        manager.append_decode_token(
            req_id, layer_name, rows[b], cur_k_nope[b : b + 1], cur_k_pe[b : b + 1]
        )
    plan = build_gather_plan(
        topk_indices, prompt_lens, block_size, manager.config.scratch_blocks_per_req
    )
    return manager.gather_decode_layer(layer_name, req_ids, plan)
