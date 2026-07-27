import argparse
import statistics

import torch

from vllm_ascend.utils import enable_custom_op

TOPK = 2048
MTP = 2
BLOCK_SIZE = 128
SELECTED_COUNT_STRIDE = 16
MAX_TOKENS = 131072
TOKEN_BASE = 4096


def _half_overlapping_rows(request_batch: int) -> torch.Tensor:
    shared = torch.arange(
        TOKEN_BASE,
        TOKEN_BASE + TOPK // 2,
        dtype=torch.int32,
        device="npu",
    )
    first_unique = torch.arange(
        TOKEN_BASE + TOPK // 2,
        TOKEN_BASE + TOPK,
        dtype=torch.int32,
        device="npu",
    )
    second_unique = torch.arange(
        TOKEN_BASE + TOPK,
        TOKEN_BASE + TOPK + TOPK // 2,
        dtype=torch.int32,
        device="npu",
    )
    request_rows = torch.stack(
        (
            torch.cat((shared, first_unique)),
            torch.cat((shared, second_unique)),
        )
    )
    return request_rows.repeat(request_batch, 1).unsqueeze(1)


def _measure_npu_ms(run, reset, warmups: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        reset()
        run()
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        reset()
        start.record()
        run()
        end.record()
    torch.npu.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _print_summary(name: str, samples: list[float]) -> float:
    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    p50 = ordered[len(ordered) // 2]
    p90 = ordered[int((len(ordered) - 1) * 0.9)]
    print(f"{name:>18}: mean={mean:.6f} ms p50={p50:.6f} ms p90={p90:.6f} ms")
    return mean


def main(request_batch: int, warmups: int, iterations: int) -> None:
    if request_batch < 1:
        raise ValueError("--request-batch must be positive")
    if warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if iterations < 1:
        raise ValueError("--iterations must be positive")
    if not enable_custom_op():
        raise RuntimeError("vllm-ascend custom operators could not be loaded")

    legacy_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_legacy_
    copy_rows_op = torch.ops._C_ascend.npu_dsa_staged_copy_rows_
    union_op = torch.ops._C_ascend.npu_dsa_staged_union_
    remap_op = torch.ops._C_ascend.npu_dsa_staged_remap_rows_

    row_count = request_batch * MTP
    request_width = MTP * TOPK
    source = _half_overlapping_rows(request_batch)
    boundaries = torch.full(
        (row_count,),
        MAX_TOKENS,
        dtype=torch.int32,
        device="npu",
    )
    valid_rows = torch.arange(row_count, dtype=torch.int32, device="npu")
    local_scratch_base = torch.zeros(
        row_count,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.arange(
        request_batch,
        dtype=torch.int32,
        device="npu",
    ).repeat_interleave(MTP)

    blocks_per_request = request_width // BLOCK_SIZE
    request_block_table = torch.arange(
        request_batch * blocks_per_request,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_batch, blocks_per_request)

    no_union_values = source.clone()
    no_union_local_indices = source.clone()
    sort_values = source.clone()
    selected_packed = torch.empty(
        (request_batch, request_width),
        dtype=torch.int32,
        device="npu",
    )
    local_to_union = torch.empty(
        (row_count, TOPK),
        dtype=torch.int32,
        device="npu",
    )
    selected_count = torch.empty(
        (request_batch, SELECTED_COUNT_STRIDE),
        dtype=torch.int32,
        device="npu",
    )
    target_slots = torch.empty(
        (request_batch, request_width),
        dtype=torch.long,
        device="npu",
    )

    def staged_no_union() -> None:
        legacy_op(
            no_union_local_indices,
            boundaries,
            valid_rows,
            local_scratch_base,
            True,
            row_requests,
        )
        copy_rows_op(no_union_values, no_union_local_indices)

    def staged_sort_union() -> None:
        row_packed = legacy_op(
            sort_values,
            boundaries,
            valid_rows,
            local_scratch_base,
            True,
            row_requests,
        )
        union_op(
            row_packed,
            selected_packed,
            local_to_union,
            selected_count,
            request_block_table,
            target_slots,
            BLOCK_SIZE,
            MAX_TOKENS,
            True,
        )
        remap_op(sort_values, local_to_union)

    staged_no_union()
    staged_sort_union()
    torch.npu.synchronize()

    expected_local_indices = torch.arange(
        TOPK,
        dtype=torch.int32,
        device="npu",
    ).expand(row_count, 1, TOPK)
    if not torch.equal(no_union_values, expected_local_indices):
        raise AssertionError("staged-no-union produced incorrect local indices")

    expected_count = TOPK + TOPK // 2
    if selected_count[:, 0].cpu().tolist() != [expected_count] * request_batch:
        raise AssertionError("staged-sort-union produced an incorrect union count")
    expected_selected = torch.arange(
        TOKEN_BASE,
        TOKEN_BASE + expected_count,
        dtype=torch.int32,
        device="npu",
    ).expand(request_batch, expected_count)
    if not torch.equal(selected_packed[:, :expected_count], expected_selected):
        raise AssertionError("staged-sort-union produced an incorrect sorted union")
    if not torch.equal(sort_values, source - TOKEN_BASE):
        raise AssertionError("staged-sort-union produced incorrect remapped rows")

    expected_targets = (
        torch.arange(
            expected_count,
            dtype=torch.long,
            device="npu",
        ).unsqueeze(0)
        + torch.arange(
            request_batch,
            dtype=torch.long,
            device="npu",
        ).unsqueeze(1)
        * request_width
    )
    if not torch.equal(target_slots[:, :expected_count], expected_targets):
        raise AssertionError("staged-sort-union produced incorrect target slots")
    print("correctness: passed")

    no_union_samples = _measure_npu_ms(
        staged_no_union,
        lambda: (
            no_union_values.copy_(source),
            no_union_local_indices.copy_(source),
        ),
        warmups,
        iterations,
    )
    sort_union_samples = _measure_npu_ms(
        staged_sort_union,
        lambda: sort_values.copy_(source),
        warmups,
        iterations,
    )

    print(f"request_batch={request_batch}, MTP={MTP}, topk={TOPK}, warmups={warmups}, iterations={iterations}")
    no_union_mean = _print_summary("staged-no-union", no_union_samples)
    sort_union_mean = _print_summary("staged-sort-union", sort_union_samples)
    delta = sort_union_mean - no_union_mean
    overhead = (sort_union_mean / no_union_mean - 1) * 100
    print(f"union overhead: {delta:+.6f} ms ({overhead:+.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original two-row staged sort union against the staged path without request-level union."
        )
    )
    parser.add_argument("--request-batch", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    main(args.request_batch, args.warmups, args.iterations)
