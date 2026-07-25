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
    if not hasattr(torch.ops._C_ascend, "npu_dsa_prepare_sparse_indices_legacy_"):
        pytest.fail("vllm_ascend_C does not contain the pre-union DSA operator")


def _buffers(requests: int, capacity: int):
    return (
        torch.empty((requests, capacity), dtype=torch.int32, device="npu"),
        torch.empty((requests, 16), dtype=torch.int32, device="npu"),
        torch.empty((requests, capacity), dtype=torch.long, device="npu"),
    )


def _aligned(values, width=16):
    result = torch.zeros((len(values), width), dtype=torch.int32)
    for row, entries in enumerate(values):
        result[row, : len(entries)] = torch.tensor(entries, dtype=torch.int32)
    return result


def test_pre_union_and_union_ops_with_half_overlapping_mtp_rows():
    topk = 2048
    shared = torch.arange(topk // 2, dtype=torch.int32)
    unique = torch.arange(topk // 2, topk + topk // 2, dtype=torch.int32)
    topk_indices = torch.stack(
        (
            torch.cat((shared, unique[: topk // 2])),
            torch.cat((shared, unique[topk // 2 :])),
        )
    ).unsqueeze(1)
    assert len(set(topk_indices[0].flatten().tolist()).intersection(topk_indices[1].flatten().tolist())) / topk == 0.5

    boundaries = torch.full((2,), 131072, dtype=torch.int32)
    row_requests = torch.zeros(2, dtype=torch.int32)
    block_size = 128
    tables = torch.arange(2 * topk // block_size, dtype=torch.int32).unsqueeze(0)

    legacy_values = topk_indices.npu()
    legacy_packed = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_legacy_(
        legacy_values,
        boundaries.npu(),
        torch.arange(2, dtype=torch.int32, device="npu"),
        torch.tensor([0, topk], dtype=torch.int32, device="npu"),
        True,
        row_requests.npu(),
    )
    union_buffers = _buffers(1, 2 * topk)
    union_values = topk_indices.npu()
    union_result = prepare_sparse_indices(
        union_values,
        boundaries.npu(),
        row_req_indices=row_requests.npu(),
        request_block_table=tables.npu(),
        selected_packed=union_buffers[0],
        selected_counts=union_buffers[1],
        target_slot_mapping=union_buffers[2],
        block_size=block_size,
    )
    torch.npu.synchronize()

    assert torch.equal(legacy_packed.cpu(), topk_indices.squeeze(1))
    assert union_result[2].cpu().tolist() == [3 * topk // 2]
    assert torch.equal(
        union_result[0][0, :, : topk // 2].cpu(),
        union_result[0][1, :, : topk // 2].cpu(),
    )


def test_mtp_rows_build_one_sorted_union_per_request():
    topk = _aligned([[3, 1, 8], [2, 3, 9], [4, 1, 10]])
    row_requests = torch.tensor([0, 0, 1], dtype=torch.int32)
    boundaries = torch.tensor([5, 5, 5], dtype=torch.int32)
    tables = _aligned([[20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35]])
    expected = _prepare_sparse_indices_torch(
        topk,
        boundaries,
        row_req_indices=row_requests,
        request_block_table=tables,
        block_size=2,
    )
    buffers = _buffers(2, 32)
    actual = prepare_sparse_indices(
        topk.npu(),
        boundaries.npu(),
        row_req_indices=row_requests.npu(),
        request_block_table=tables.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )
    torch.npu.synchronize()

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[2].cpu(), expected[2])
    for request, count in enumerate(expected[2].tolist()):
        assert torch.equal(
            actual[1][request, :count].cpu(),
            expected[1][request, :count],
        )
        assert torch.equal(
            actual[3][request, :count].cpu(),
            expected[3][request, :count],
        )


def test_q1_requests_remain_independent():
    topk = _aligned([[1, 3, 8], [1, 4, 9]])
    row_requests = torch.tensor([0, 1], dtype=torch.int32)
    boundaries = torch.tensor([5, 5], dtype=torch.int32)
    tables = _aligned([[20, 21, 22], [30, 31, 32]])
    expected = _prepare_sparse_indices_torch(
        topk,
        boundaries,
        row_req_indices=row_requests,
        request_block_table=tables,
        block_size=2,
    )
    buffers = _buffers(2, 16)
    actual = prepare_sparse_indices(
        topk.npu(),
        boundaries.npu(),
        row_req_indices=row_requests.npu(),
        request_block_table=tables.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )
    torch.npu.synchronize()
    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[2].cpu(), expected[2])


def test_zero_boundary_keeps_resident_absolute_indices():
    topk = _aligned([[1, 3, 8], [2, 4, 9]])
    original = topk.clone()
    row_requests = torch.tensor([0, 0], dtype=torch.int32)
    tables = _aligned([[20, 21, 22]])
    buffers = _buffers(1, 32)
    actual = prepare_sparse_indices(
        topk.npu(),
        torch.zeros(2, dtype=torch.int32, device="npu"),
        row_req_indices=row_requests.npu(),
        request_block_table=tables.npu(),
        selected_packed=buffers[0],
        selected_counts=buffers[1],
        target_slot_mapping=buffers[2],
        block_size=2,
    )
    torch.npu.synchronize()
    assert torch.equal(actual[0].cpu(), original)
    assert actual[2].cpu().tolist() == [0]
