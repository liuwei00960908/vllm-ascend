import time

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    prepare_sparse_indices,
)


def main(rows=4, requests=2, topk=2048, iterations=200):
    values = torch.randint(0, 131072, (rows, 1, topk), dtype=torch.int32, device="npu")
    boundaries = torch.full((rows,), 131072, dtype=torch.int32, device="npu")
    row_requests = torch.arange(rows, dtype=torch.int32, device="npu") % requests
    block_table = torch.arange(
        requests * 1024, dtype=torch.int32, device="npu"
    ).reshape(requests, 1024)
    selected = torch.empty((requests, rows * topk), dtype=torch.int32, device="npu")
    counts = torch.empty((requests, 16), dtype=torch.int32, device="npu")
    targets = torch.empty((requests, rows * topk), dtype=torch.long, device="npu")

    def run():
        prepare_sparse_indices(
            values,
            boundaries,
            row_req_indices=row_requests,
            request_block_table=block_table,
            selected_packed=selected,
            selected_counts=counts,
            target_slot_mapping=targets,
            block_size=128,
        )

    for _ in range(20):
        run()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        run()
    torch.npu.synchronize()
    print(f"{(time.perf_counter() - started) * 1000 / iterations:.6f} ms")


if __name__ == "__main__":
    main()
