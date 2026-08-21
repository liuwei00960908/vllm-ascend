"""Device-only sparse-index preparation for DSA latent scratch (B2e).

Decode reads the latent through two disjoint index spaces resolved by the
SAME per-request block table:

  * LMCache-selected positions (< cache boundary) -> request-level bitmap
    union scratch rows [0..n_unique), ordered by absolute token position
    and shared by all MTP rows for that request;
  * live-cache positions (>= cache boundary) -> kept ABSOLUTE, read in
    place from their tail blocks. No copy, no [retrieve|decode] assembly.

A zero boundary selects nothing from LMCache and leaves every index
absolute. Everything is fixed-shape tensor math: no D2H sync.

Provenance: vllm-ascend-sparse@c7c4a4ac
distributed/kv_transfer/sparse_offload/prepare_sparse_indices.py (whole
file), adapted for the eager replay slice:

* only the NORMAL kernel variant is wired (``npu_dsa_prepare_sparse_
  indices_``); the staged variant + its fixed-address workspace contract
  belong to the P9 staged-graph slice;
* output buffers are allocated here instead of being caller-owned
  fixed-address storage (that requirement exists for graph replay only);
  ``_prepare_sparse_indices_torch`` is kept verbatim as the test oracle.
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
):
    """Request-level sorted bitmap-union reference used as a test oracle."""
    orig_shape = topk_indices.shape
    sel = topk_indices.reshape(orig_shape[0], -1)
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
        selected_tokens = sorted(
            {
                int(sel[row, col])
                for row in range(sel.shape[0])
                if int(row_req_indices[row]) == req
                for col in range(sel.shape[1])
                if 0 <= int(sel[row, col]) < int(split_boundary[row])
            }
        )
        inverse = {token: slot for slot, token in enumerate(selected_tokens)}
        for token, slot in inverse.items():
            packed[req, slot] = token
            block_id = int(request_block_table[req, slot // block_size])
            targets[req, slot] = block_id * block_size + slot % block_size
        for row in range(sel.shape[0]):
            if int(row_req_indices[row]) != req:
                continue
            boundary = int(split_boundary[row])
            for col in range(sel.shape[1]):
                token = int(sel[row, col])
                if 0 <= token < boundary:
                    new_indices[row, col] = inverse[token]
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
    request_block_table: torch.Tensor,
    block_size: int,
    device: torch.device,
    row_req_indices: torch.Tensor | None = None,
    scratch_capacity: int | None = None,
    clear_invalid_rows: bool = False,
):
    """Remap absolute top-k indices for the compact-scratch decode path.

    Args:
        topk_indices: [rows, k] absolute token positions selected by the
            indexer (padding entries are non-positive).
        split_boundary: [rows] per-row cache boundary. Zero means the whole
            prefix is resident (nothing selected); a positive value is the
            LMCache-committed frontier and positions below it are remapped
            through the request-level union scratch.
        request_block_table: [num_requests, blocks] the latent group's block
            table (scratch slots resolve through it).
        block_size: KV block size (must divide index_topk).
        device: NPU device for the output buffers.
        row_req_indices: [rows] request index per row; negative entries are
            padding rows (zeroed).
        scratch_capacity: per-request capacity of the output buffers;
            defaults to topk per row (one request's union upper bound).

    Returns:
        new_indices: same shape as topk_indices — selected entries replaced
            by their compact scratch row, live/padding entries unchanged.
        selected_packed: [num_requests, capacity] int32 deduplicated
            selected token list (0-padded).
        selected_counts: [num_requests] int32 per-request selected count.
        target_slot_mapping: [num_requests, capacity] int64 physical slots
            (block_table[j // block_size] * block_size + j % block_size).
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

    rows = int(topk_indices.shape[0])
    request_count = int(request_block_table.shape[0])
    if row_req_indices is None:
        row_req_indices = torch.arange(
            rows, dtype=torch.int32, device=topk_indices.device
        )[:request_count].repeat(
            1 + (rows - 1) // max(request_count, 1)
        )[:rows].to(torch.int32)
    if scratch_capacity is None:
        scratch_capacity = int(
            topk_indices.reshape(rows, -1).shape[1]
        )

    selected_packed = torch.zeros(
        (request_count, scratch_capacity), dtype=torch.int32, device=device
    )
    selected_counts = torch.zeros(
        (request_count, 16), dtype=torch.int32, device=device
    )
    target_slot_mapping = torch.zeros(
        (request_count, scratch_capacity), dtype=torch.long, device=device
    )
    fused_op(
        topk_indices,
        split_boundary,
        row_req_indices,
        request_block_table,
        selected_packed,
        selected_counts,
        target_slot_mapping,
        block_size,
        True,
        clear_invalid_rows,
    )
    return (
        topk_indices,
        selected_packed,
        selected_counts[:, 0],
        target_slot_mapping,
    )
