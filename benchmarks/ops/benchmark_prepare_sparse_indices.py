import argparse
import statistics

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    prepare_sparse_indices,
)
from vllm_ascend.utils import enable_custom_op


def _two_rows_with_half_overlap(topk: int, device: str) -> torch.Tensor:
    if topk % 2:
        raise ValueError("topk must be even for an exact 0.5 row overlap")
    shared = torch.arange(topk // 2, dtype=torch.int32, device=device)
    unique = torch.arange(topk // 2, topk + topk // 2, dtype=torch.int32, device=device)
    first = torch.cat((shared, unique[: topk // 2]))
    second = torch.cat((shared, unique[topk // 2 :]))
    return torch.stack((first, second)).unsqueeze(1)


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


def main(topk: int = 2048, iterations: int = 200, warmups: int = 20) -> None:
    if not enable_custom_op():
        raise RuntimeError("vllm-ascend custom operators could not be loaded")
    legacy_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_legacy_

    source = _two_rows_with_half_overlap(topk, "npu")
    legacy_values = source.clone()
    union_values = source.clone()
    boundaries = torch.full((2,), 131072, dtype=torch.int32, device="npu")
    row_requests = torch.zeros(2, dtype=torch.int32, device="npu")

    # The pre-union operator assigns each row its own compact scratch range.
    valid_rows = torch.arange(2, dtype=torch.int32, device="npu")
    scratch_base = torch.arange(2, dtype=torch.int32, device="npu") * topk

    block_size = 128
    capacity = 2 * topk
    block_table = torch.arange(capacity // block_size, dtype=torch.int32, device="npu").unsqueeze(0)
    selected = torch.empty((1, capacity), dtype=torch.int32, device="npu")
    counts = torch.empty((1, 16), dtype=torch.int32, device="npu")
    targets = torch.empty((1, capacity), dtype=torch.long, device="npu")

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

    legacy_mean = statistics.fmean(legacy_samples)
    union_mean = statistics.fmean(union_samples)
    _summary("pre-union", legacy_samples)
    _summary("union", union_samples)
    print(f"union overhead: {union_mean - legacy_mean:+.6f} ms ({(union_mean / legacy_mean - 1) * 100:+.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=20)
    args = parser.parse_args()
    main(args.topk, args.iterations, args.warmups)
