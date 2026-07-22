# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    _prepare_sparse_indices_torch,
)
from vllm_ascend.utils import enable_custom_op


def _measure(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--mtp-rows", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=170000)
    parser.add_argument("--boundary", type=int, default=131584)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if min(args.requests, args.mtp_rows, args.topk, args.max_model_len) <= 0:
        parser.error("requests, mtp-rows, topk and max-model-len must be positive")
    if not 0 <= args.boundary <= args.max_model_len:
        parser.error("boundary must be in [0, max-model-len]")

    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError as exc:
        raise RuntimeError(
            "vllm_ascend_C does not contain "
            "npu_dsa_prepare_sparse_indices_; rebuild the extension before "
            "running this benchmark"
        ) from exc

    device = torch.device("npu")
    rows = args.requests * args.mtp_rows
    scratch_capacity = args.mtp_rows * args.topk
    topk = torch.randint(
        0,
        args.max_model_len,
        (rows, args.topk),
        dtype=torch.int32,
        device=device,
    )
    split_boundary = torch.full(
        (rows,), args.boundary, dtype=torch.int32, device=device
    )
    row_req_indices = torch.arange(
        args.requests, dtype=torch.int32, device=device
    ).repeat_interleave(args.mtp_rows)
    block_table_width = (
        args.max_model_len + args.block_size - 1
    ) // args.block_size
    request_block_table = torch.arange(
        args.requests * block_table_width,
        dtype=torch.int32,
        device=device,
    ).view(args.requests, block_table_width)
    selected = torch.empty(
        (args.requests, scratch_capacity), dtype=torch.int32, device=device
    )
    counts = torch.empty(args.requests, dtype=torch.int32, device=device)
    targets = torch.empty(
        (args.requests, scratch_capacity), dtype=torch.long, device=device
    )
    expected = _prepare_sparse_indices_torch(
        topk.cpu(),
        split_boundary.cpu(),
        row_req_indices=row_req_indices.cpu(),
        request_block_table=request_block_table.cpu(),
        block_size=args.block_size,
    )
    actual_topk = topk.clone()
    fused_op(
        actual_topk,
        split_boundary,
        row_req_indices,
        request_block_table,
        selected,
        counts,
        targets,
        args.block_size,
        True,
        False,
    )
    torch.npu.synchronize()
    expected_topk, expected_selected, expected_counts, expected_targets = expected
    if not torch.equal(actual_topk.cpu(), expected_topk):
        raise AssertionError("bitmap remap differs from the Torch reference")
    if not torch.equal(counts.cpu(), expected_counts):
        raise AssertionError("bitmap union counts differ from the Torch reference")
    for req, count in enumerate(expected_counts.tolist()):
        if not torch.equal(selected[req, :count].cpu(), expected_selected[req, :count]):
            raise AssertionError("bitmap union payload differs from the Torch reference")
        if not torch.equal(targets[req, :count].cpu(), expected_targets[req, :count]):
            raise AssertionError("bitmap target slots differ from the Torch reference")

    fused_topk = torch.empty_like(topk)

    def fused() -> None:
        # The operator remaps in place, so reset the fixed input every iteration.
        fused_topk.copy_(topk)
        fused_op(
            fused_topk,
            split_boundary,
            row_req_indices,
            request_block_table,
            selected,
            counts,
            targets,
            args.block_size,
            True,
            False,
        )

    fused_ms = _measure(fused, args.warmup, args.iterations)
    print(
        f"requests={args.requests} mtp_rows={args.mtp_rows} topk={args.topk} "
        f"max_model_len={args.max_model_len} boundary={args.boundary} "
        f"exact_match=True fused_with_input_copy_ms={fused_ms:.6f}"
    )


if __name__ == "__main__":
    main()
