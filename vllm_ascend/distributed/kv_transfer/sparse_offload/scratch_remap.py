"""Device-only top-k remap for the DSA latent scratch (Step B2).

Decode reads the latent through two disjoint index spaces resolved by the SAME
per-request block table:

  * LMCache-selected positions (< cache boundary) -> compact scratch rows [0..n_ret)
    (the request's first ceil(k/block_size) latent blocks, filled by LMCache);
  * live-cache positions (>= cache boundary >= k) -> kept ABSOLUTE, read in
    place from their tail blocks. No copy, no [retrieve|decode] assembly.

Everything is fixed-shape tensor math: no D2H sync, graph-mode friendly.
"""

import torch


def _scratch_remap_torch(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    need_packed: bool = True,
    scratch_base: torch.Tensor | None = None,
    valid_row_indices: torch.Tensor | None = None,
):
    """Original Torch implementation, retained only as a test oracle."""
    orig_shape = topk_indices.shape
    sel = topk_indices.reshape(orig_shape[0], -1)
    k = sel.shape[1]
    boundary = split_boundary.reshape(-1, 1).to(device=sel.device, dtype=sel.dtype)
    if scratch_base is None:
        base = torch.zeros((sel.shape[0], 1), dtype=sel.dtype, device=sel.device)
    else:
        base = scratch_base.reshape(-1, 1).to(device=sel.device, dtype=sel.dtype)

    is_lmcache = (sel >= 0) & (sel < boundary)
    # torch.cumsum promotes integer dtypes to int64 by default; the sparse FA
    # kernel requires int32 indices, so pin the dtype explicitly.
    rank = torch.cumsum(is_lmcache, dim=1, dtype=sel.dtype) - 1
    new_indices = torch.where(is_lmcache, base + rank, sel)

    if not need_packed:
        return new_indices.reshape(orig_shape), None

    # Add one trash column so non-LMCache entries scatter harmlessly off the
    # end, then return only the front-packed absolute positions.
    packed = sel.new_zeros(sel.shape[0], k + 1)
    dst = torch.where(is_lmcache, rank, torch.full_like(rank, k))
    packed.scatter_(1, dst.long(), sel)
    packed = packed[:, :k].to(torch.int32)
    if valid_row_indices is not None:
        packed = packed.index_select(
            0,
            valid_row_indices.reshape(-1).to(device=sel.device, dtype=torch.long),
        )
    return new_indices.reshape(orig_shape), packed


def scratch_remap(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    need_packed: bool = True,
    scratch_base: torch.Tensor | None = None,
    valid_row_indices: torch.Tensor | None = None,
):
    """Remap absolute top-k indices for the compact-scratch decode path.

    Args:
        topk_indices: [bs, 1, k] (or [bs, k]) absolute token positions selected
            by the indexer; negative entries are padding.
        split_boundary: [bs] cache split boundary per decode request. In the original
            mode this is the prompt length; decode-window mode passes the
            current window start. Callers must ensure boundary >= k for every
            row (else scratch rows would alias live-cache positions).
        need_packed: whether to build the LMCache selected-token payload.
        scratch_base: [bs] compact scratch base per row. This lets MTP
            rows for the same request use disjoint compact scratch ranges.
        valid_row_indices: [num_decode_rows] ordered, unique source-row
            indices. Only those rows are remapped; selected_packed follows this
            order with shape [num_decode_rows, k]. The custom op mutates these
            topk_indices rows in place.

    Returns:
        new_indices: same shape as topk_indices. LMCache-selected entries are
            replaced by their compact scratch row (scratch_base + rank in
            top-k order); live-cache and padding entries stay unchanged.
        selected_packed: [num_decode_rows, k] int32. LMCache-selected ABSOLUTE
            positions are front-packed in top-k order (row i goes to scratch
            slot i), with the tail padded with 0. None when need_packed=False.
    """
    if topk_indices.device.type != "npu":
        raise RuntimeError(
            "scratch_remap requires the NPU custom op; use _scratch_remap_torch only as a test reference"
        )
    if valid_row_indices is None or scratch_base is None:
        raise ValueError("scratch_remap requires valid_row_indices and scratch_base")

    try:
        fused_op = torch.ops._C_ascend.npu_dsa_scratch_remap_
    except AttributeError as exc:
        raise RuntimeError(
            "vllm_ascend_C does not expose npu_dsa_scratch_remap_; rebuild the custom-op extension"
        ) from exc

    packed = fused_op(
        topk_indices,
        split_boundary,
        valid_row_indices,
        scratch_base,
        need_packed,
    )
    return topk_indices, packed if need_packed else None
