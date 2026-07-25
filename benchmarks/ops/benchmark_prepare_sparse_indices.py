import argparse
import statistics

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    prepare_sparse_indices,
)
from vllm_ascend.utils import enable_custom_op


def _mtp_rows_with_half_overlap(
    topk: int, request_batch: int, device: str
) -> torch.Tensor:
    if topk % 2:
        raise ValueError("topk must be even for an exact 0.5 row overlap")
    shared = torch.arange(topk // 2, dtype=torch.int32, device=device)
    unique = torch.arange(topk // 2, topk + topk // 2, dtype=torch.int32, device=device)
    first = torch.cat((shared, unique[: topk // 2]))
    second = torch.cat((shared, unique[topk // 2 :]))
    request_rows = torch.stack((first, second))
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


def _summary(name: str, samples: list[float]) -> None:
    ordered = sorted(samples)
    p50 = ordered[len(ordered) // 2]
    p90 = ordered[int((len(ordered) - 1) * 0.9)]
    print(f"{name:>12}: mean={statistics.fmean(samples):.6f} ms p50={p50:.6f} ms p90={p90:.6f} ms")


def _staged_runner(
    *,
    legacy_op,
    union_op,
    remap_op,
    values,
    boundaries,
    valid_rows,
    local_scratch_base,
    row_requests,
    request_block_table,
    selected_packed,
    local_to_union,
    selected_count,
    target_slots,
    block_size,
    max_tokens,
    use_sort,
):
    row_packed = legacy_op(
        values,
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
        block_size,
        max_tokens,
        use_sort,
    )
    remap_op(values, local_to_union)
    return values, selected_packed, selected_count, target_slots


def _staged_no_union_runner(
    *,
    legacy_op,
    copy_rows_op,
    values,
    local_indices,
    boundaries,
    valid_rows,
    local_scratch_base,
    row_requests,
):
    legacy_op(
        local_indices,
        boundaries,
        valid_rows,
        local_scratch_base,
        True,
        row_requests,
    )
    copy_rows_op(values, local_indices)
    return values


def main(topk: int = 2048, iterations: int = 200, warmups: int = 20) -> None:
    if topk != 2048:
        raise ValueError("the experimental staged kernels currently require --topk 2048")
    if not enable_custom_op():
        raise RuntimeError("vllm-ascend custom operators could not be loaded")
    legacy_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_legacy_
    union_op = torch.ops._C_ascend.npu_dsa_staged_union_
    remap_op = torch.ops._C_ascend.npu_dsa_staged_remap_rows_
    copy_rows_op = torch.ops._C_ascend.npu_dsa_staged_copy_rows_

    request_batch = 4
    row_count = 2 * request_batch
    source = _mtp_rows_with_half_overlap(topk, request_batch, "npu")
    legacy_values = source.clone()
    union_values = source.clone()
    no_union_values = source.clone()
    no_union_local_indices = source.clone()
    bitmap_values = source.clone()
    sort_values = source.clone()
    max_tokens = 131072
    boundaries = torch.full(
        (row_count,), max_tokens, dtype=torch.int32, device="npu"
    )
    row_requests = torch.arange(
        request_batch, dtype=torch.int32, device="npu"
    ).repeat_interleave(2)

    # The pre-union operator assigns each row its own compact scratch range.
    valid_rows = torch.arange(row_count, dtype=torch.int32, device="npu")
    scratch_base = (
        torch.arange(row_count, dtype=torch.int32, device="npu") * topk
    )
    local_scratch_base = torch.zeros(
        row_count, dtype=torch.int32, device="npu"
    )

    block_size = 128
    capacity = 2 * topk
    blocks_per_request = max_tokens // block_size
    block_table = torch.arange(
        request_batch * blocks_per_request,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_batch, blocks_per_request)
    selected = torch.empty(
        (request_batch, capacity), dtype=torch.int32, device="npu"
    )
    counts = torch.empty(
        (request_batch, 16), dtype=torch.int32, device="npu"
    )
    targets = torch.empty(
        (request_batch, capacity), dtype=torch.long, device="npu"
    )
    bitmap_buffers = (
        torch.empty(
            (request_batch, capacity), dtype=torch.int32, device="npu"
        ),
        torch.empty((row_count, topk), dtype=torch.int32, device="npu"),
        torch.empty(
            (request_batch, 16), dtype=torch.int32, device="npu"
        ),
        torch.empty(
            (request_batch, capacity), dtype=torch.long, device="npu"
        ),
    )
    sort_buffers = tuple(torch.empty_like(item) for item in bitmap_buffers)

    def staged(values, buffers, use_sort):
        return _staged_runner(
            legacy_op=legacy_op,
            union_op=union_op,
            remap_op=remap_op,
            values=values,
            boundaries=boundaries,
            valid_rows=valid_rows,
            local_scratch_base=local_scratch_base,
            row_requests=row_requests,
            request_block_table=block_table,
            selected_packed=buffers[0],
            local_to_union=buffers[1],
            selected_count=buffers[2],
            target_slots=buffers[3],
            block_size=block_size,
            max_tokens=max_tokens,
            use_sort=use_sort,
        )

    def staged_no_union():
        return _staged_no_union_runner(
            legacy_op=legacy_op,
            copy_rows_op=copy_rows_op,
            values=no_union_values,
            local_indices=no_union_local_indices,
            boundaries=boundaries,
            valid_rows=valid_rows,
            local_scratch_base=local_scratch_base,
            row_requests=row_requests,
        )

    no_union_result = staged_no_union()
    bitmap_result = staged(bitmap_values, bitmap_buffers, False)
    sort_result = staged(sort_values, sort_buffers, True)
    torch.npu.synchronize()
    expected_local_indices = torch.arange(
        topk, dtype=torch.int32, device="npu"
    ).expand(row_count, 1, -1)
    if not torch.equal(no_union_result.cpu(), expected_local_indices.cpu()):
        raise AssertionError("staged no-union remapped rows are incorrect")
    expected_count = 3 * topk // 2
    expected_counts = [expected_count] * request_batch
    if bitmap_result[2][:, 0].cpu().tolist() != expected_counts:
        raise AssertionError("bitmap staged union count is incorrect")
    if sort_result[2][:, 0].cpu().tolist() != expected_counts:
        raise AssertionError("sort staged union count is incorrect")
    if not torch.equal(bitmap_result[0].cpu(), sort_result[0].cpu()):
        raise AssertionError("bitmap and sort remapped rows differ")
    for index in (1, 3):
        if not torch.equal(
            bitmap_result[index][:, :expected_count].cpu(),
            sort_result[index][:, :expected_count].cpu(),
        ):
            raise AssertionError("bitmap and sort staged payloads differ")

    legacy_samples = _measure_npu_ms(
        lambda: legacy_op(
            legacy_values,
            boundaries,
            valid_rows,
            scratch_base,
            True,
            row_requests,
        ),
        lambda: legacy_values.copy_(source),
        warmups,
        iterations,
    )
    union_samples = _measure_npu_ms(
        lambda: prepare_sparse_indices(
            union_values,
            boundaries,
            row_req_indices=row_requests,
            request_block_table=block_table,
            selected_packed=selected,
            selected_counts=counts,
            target_slot_mapping=targets,
            block_size=block_size,
        ),
        lambda: union_values.copy_(source),
        warmups,
        iterations,
    )
    no_union_samples = _measure_npu_ms(
        staged_no_union,
        lambda: no_union_local_indices.copy_(source),
        warmups,
        iterations,
    )
    bitmap_samples = _measure_npu_ms(
        lambda: staged(bitmap_values, bitmap_buffers, False),
        lambda: bitmap_values.copy_(source),
        warmups,
        iterations,
    )
    sort_samples = _measure_npu_ms(
        lambda: staged(sort_values, sort_buffers, True),
        lambda: sort_values.copy_(source),
        warmups,
        iterations,
    )

    legacy_mean = statistics.fmean(legacy_samples)
    union_mean = statistics.fmean(union_samples)
    no_union_mean = statistics.fmean(no_union_samples)
    bitmap_mean = statistics.fmean(bitmap_samples)
    sort_mean = statistics.fmean(sort_samples)
    _summary("pre-union", legacy_samples)
    _summary("fused-union", union_samples)
    _summary("staged-no-union", no_union_samples)
    _summary("staged-bitmap", bitmap_samples)
    _summary("staged-sort", sort_samples)
    print(f"union overhead: {union_mean - legacy_mean:+.6f} ms ({(union_mean / legacy_mean - 1) * 100:+.2f}%)")
    print(
        "staged bitmap union cost: "
        f"{bitmap_mean - no_union_mean:+.6f} ms "
        f"({(bitmap_mean / no_union_mean - 1) * 100:+.2f}%)"
    )
    print(
        "staged sort union cost: "
        f"{sort_mean - no_union_mean:+.6f} ms "
        f"({(sort_mean / no_union_mean - 1) * 100:+.2f}%)"
    )
    fastest = min(
        (
            ("pre-union", legacy_mean),
            ("fused-union", union_mean),
            ("staged-no-union", no_union_mean),
            ("staged-bitmap", bitmap_mean),
            ("staged-sort", sort_mean),
        ),
        key=lambda item: item[1],
    )
    print(f"fastest: {fastest[0]} ({fastest[1]:.6f} ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=20)
    args = parser.parse_args()
    main(args.topk, args.iterations, args.warmups)
