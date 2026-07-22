import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    _prepare_sparse_indices_torch,
    prepare_sparse_indices,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def _load_dsa_union_operator():
    if not enable_custom_op():
        pytest.fail("vllm-ascend custom operators could not be loaded")
    if not hasattr(torch.ops._C_ascend, "npu_dsa_prepare_sparse_indices_"):
        pytest.fail("vllm_ascend_C does not contain the DSA union operator")


def _buffers(requests: int, capacity: int):
    return (
        torch.empty((requests, capacity), dtype=torch.int32, device="npu"),
        torch.empty(requests, dtype=torch.int32, device="npu"),
        torch.empty((requests, capacity), dtype=torch.long, device="npu"),
    )


@pytest.mark.parametrize("boundary", [0, 2, 4, 8])
def test_request_union_matches_cpu_reference_and_preserves_live_indices(boundary):
    topk_cpu = torch.tensor(
        [[0, 1, 2, 7], [1, 0, 3, 8]], dtype=torch.int32
    )
    row_req_cpu = torch.tensor([0, 0], dtype=torch.int32)
    table_cpu = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32)
    expected, packed, counts, targets = _prepare_sparse_indices_torch(
        topk_cpu,
        torch.tensor([boundary, boundary], dtype=torch.int32),
        row_req_indices=row_req_cpu,
        request_block_table=table_cpu,
        block_size=2,
    )

    actual = topk_cpu.npu()
    buffers = _buffers(1, 8)
    actual, actual_packed, actual_counts, actual_targets = prepare_sparse_indices(
        actual,
        torch.tensor([boundary, boundary], dtype=torch.int32, device="npu"),
        row_req_indices=row_req_cpu.npu(),
        request_block_table=table_cpu.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )

    count = int(counts.item())
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(actual_counts.cpu(), counts)
    assert torch.equal(actual_packed[0, :count].cpu(), packed[0, :count])
    assert torch.equal(actual_targets[0, :count].cpu(), targets[0, :count])


def test_two_requests_are_deduplicated_independently():
    topk_cpu = torch.tensor(
        [[1, 2, 9, 10], [2, 3, 10, 11], [1, 4, 12, 13]],
        dtype=torch.int32,
    )
    row_req_cpu = torch.tensor([0, 0, 1], dtype=torch.int32)
    table_cpu = torch.tensor(
        [[20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35]],
        dtype=torch.int32,
    )
    expected, packed, counts, targets = _prepare_sparse_indices_torch(
        topk_cpu,
        torch.tensor([4, 4, 5], dtype=torch.int32),
        row_req_indices=row_req_cpu,
        request_block_table=table_cpu,
        block_size=2,
    )
    buffers = _buffers(2, 8)
    actual, actual_packed, actual_counts, actual_targets = prepare_sparse_indices(
        topk_cpu.npu(),
        torch.tensor([4, 4, 5], dtype=torch.int32, device="npu"),
        row_req_indices=row_req_cpu.npu(),
        request_block_table=table_cpu.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(actual_counts.cpu(), counts)
    for req, count in enumerate(counts.tolist()):
        assert torch.equal(actual_packed[req, :count].cpu(), packed[req, :count])
        assert torch.equal(actual_targets[req, :count].cpu(), targets[req, :count])


def test_bitmap_union_assigns_sorted_position_ranks():
    topk_cpu = torch.tensor([[3, 1], [2, 3]], dtype=torch.int32)
    row_req_cpu = torch.tensor([0, 0], dtype=torch.int32)
    table_cpu = torch.tensor([[20, 21]], dtype=torch.int32)
    expected, packed, counts, targets = _prepare_sparse_indices_torch(
        topk_cpu,
        torch.tensor([4, 4], dtype=torch.int32),
        row_req_indices=row_req_cpu,
        request_block_table=table_cpu,
        block_size=2,
    )
    buffers = _buffers(1, 4)
    actual, actual_packed, actual_counts, actual_targets = prepare_sparse_indices(
        topk_cpu.npu(),
        torch.tensor([4, 4], dtype=torch.int32, device="npu"),
        row_req_indices=row_req_cpu.npu(),
        request_block_table=table_cpu.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )

    assert packed[0, :3].tolist() == [1, 2, 3]
    assert expected.tolist() == [[2, 0], [1, 2]]
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(actual_counts.cpu(), counts)
    assert torch.equal(actual_packed[0, :3].cpu(), packed[0, :3])
    assert torch.equal(actual_targets[0, :3].cpu(), targets[0, :3])
