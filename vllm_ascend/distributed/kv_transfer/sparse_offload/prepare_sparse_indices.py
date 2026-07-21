"""Device-only sparse-index preparation for DSA latent scratch (Step B2).

Decode reads the latent through two disjoint index spaces resolved by the SAME
per-request block table:

  * LMCache-selected positions (< cache boundary) -> compact scratch rows [0..n_ret)
    (the request's first ceil(k/block_size) latent blocks, filled by LMCache);
  * live-cache positions (>= cache boundary >= k) -> kept ABSOLUTE, read in
    place from their tail blocks. No copy, no [retrieve|decode] assembly.

Everything is fixed-shape tensor math: no D2H sync, graph-mode friendly.
"""

import torch


def _prepare_sparse_indices_torch(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    row_req_indices: torch.Tensor | None = None,
    request_block_table: torch.Tensor | None = None,
    block_size: int = 1,
    need_packed: bool = True,
    clear_invalid_rows: bool = False,
    scratch_base: torch.Tensor | None = None,
    valid_row_indices: torch.Tensor | None = None,
):
    """Request-level stable-union reference, retained only as a test oracle."""
    orig_shape = topk_indices.shape
    sel = topk_indices.reshape(orig_shape[0], -1)
    if request_block_table is None:
        boundary = split_boundary.reshape(-1, 1).to(sel)
        base = (
            torch.zeros((sel.shape[0], 1), dtype=sel.dtype, device=sel.device)
            if scratch_base is None
            else scratch_base.reshape(-1, 1).to(sel)
        )
        selected = (sel >= 0) & (sel < boundary)
        rank = torch.cumsum(selected, dim=1, dtype=sel.dtype) - 1
        remapped = torch.where(selected, base + rank, sel)
        if row_req_indices is not None:
            remapped[row_req_indices[: sel.shape[0]] < 0] = 0
        if not need_packed:
            return remapped.reshape(orig_shape), None
        packed = sel.new_zeros((sel.shape[0], sel.shape[1] + 1))
        dst = torch.where(selected, rank, torch.full_like(rank, sel.shape[1]))
        packed.scatter_(1, dst.long(), sel)
        packed = packed[:, : sel.shape[1]]
        if valid_row_indices is not None:
            packed = packed.index_select(0, valid_row_indices.long())
        return remapped.reshape(orig_shape), packed

    assert row_req_indices is not None
    request_count = int(request_block_table.shape[0])
    capacity = sel.shape[1] * max(
        1,
        max(
            (
                int((row_req_indices == req).sum())
                for req in range(request_count)
            ),
            default=1,
        ),
    )
    packed = sel.new_zeros((request_count, capacity))
    counts = torch.zeros(request_count, dtype=torch.int32, device=sel.device)
    targets = torch.zeros(
        (request_count, capacity), dtype=torch.long, device=sel.device
    )
    new_indices = sel.clone()
    for req in range(request_count):
        inverse: dict[int, int] = {}
        for row in range(sel.shape[0]):
            if int(row_req_indices[row]) != req:
                continue
            boundary = int(split_boundary[row])
            for col in range(sel.shape[1]):
                token = int(sel[row, col])
                if 0 <= token < boundary:
                    slot = inverse.get(token)
                    if slot is None:
                        slot = len(inverse)
                        inverse[token] = slot
                        packed[req, slot] = token
                        block_id = int(request_block_table[req, slot // block_size])
                        targets[req, slot] = (
                            block_id * block_size + slot % block_size
                        )
                    new_indices[row, col] = slot
        counts[req] = len(inverse)
    if clear_invalid_rows:
        new_indices[row_req_indices[: sel.shape[0]] < 0] = 0
    return (
        new_indices.reshape(orig_shape),
        packed if need_packed else None,
        counts if need_packed else None,
        targets if need_packed else None,
    )


def prepare_sparse_indices(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    row_req_indices: torch.Tensor,
    request_block_table: torch.Tensor,
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slot_mapping: torch.Tensor,
    hash_workspace: torch.Tensor,
    block_size: int,
    need_packed: bool = True,
    clear_invalid_rows: bool = False,
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
        row_req_indices: [bs] request index for each row; negative entries are
            zeroed in the same kernel. Pass this only for pure
            decode/spec-decode; a mixed prefill row also has a negative request
            index but is real.

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
            "prepare_sparse_indices requires the NPU custom op; use "
            "_prepare_sparse_indices_torch only as a test reference"
        )
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError as exc:
        raise RuntimeError(
            "vllm_ascend_C does not expose npu_dsa_prepare_sparse_indices_; "
            "rebuild the custom-op extension"
        ) from exc

    fused_op(
        topk_indices,
        split_boundary,
        row_req_indices,
        request_block_table,
        selected_packed,
        selected_counts,
        target_slot_mapping,
        hash_workspace,
        block_size,
        need_packed,
        clear_invalid_rows,
    )
    return (
        topk_indices,
        selected_packed if need_packed else None,
        selected_counts if need_packed else None,
        target_slot_mapping if need_packed else None,
    )
