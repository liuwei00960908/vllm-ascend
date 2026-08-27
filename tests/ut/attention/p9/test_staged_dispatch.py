#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unit tests for the staged sparse-index dispatch (P9 batch 1). The
# parameter-contract cases run on any device; the semantic cases pin the
# staged Q1 layout (staged_mtp=1: one top-k row per request) source-order
# layout against the sorted oracle so the two
# orderings can never drift apart. The NPU kernel comparison itself runs
# on the test machine (dev boxes have no NPU runtime).
#

import unittest

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (  # noqa: E501
    _prepare_sparse_indices_torch,
    prepare_sparse_indices,
)


def _staged_mtp1_reference(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    request_block_table: torch.Tensor,
    block_size: int,
):
    """CPU reference for the staged Q1-layout kernel semantics.

    One row per request: selected (< boundary) tokens keep their top-k
    source order, are compacted to row-local ranks, and the same ranks are
    written back into the indices. Duplicated tokens within one row keep
    first-occurrence rank. Unselected entries stay absolute.
    """
    rows, width = topk_indices.shape
    request_count = request_block_table.shape[0]
    assert rows == request_count
    new_indices = topk_indices.clone()
    packed = torch.zeros((request_count, width), dtype=torch.int32)
    counts = torch.zeros(request_count, dtype=torch.int32)
    targets = torch.zeros((request_count, width), dtype=torch.int64)
    for row in range(rows):
        boundary = int(split_boundary[row])
        if boundary <= 0:
            counts[row] = 0
            continue
        order = {}
        for col in range(width):
            token = int(topk_indices[row, col])
            if 0 <= token < boundary and token not in order:
                order[token] = len(order)
        for token, rank in order.items():
            packed[row, rank] = token
            block_id = int(request_block_table[row, rank // block_size])
            targets[row, rank] = block_id * block_size + rank % block_size
        for col in range(width):
            token = int(topk_indices[row, col])
            if 0 <= token < boundary:
                new_indices[row, col] = order[token]
        counts[row] = len(order)
    return new_indices, packed, counts, targets


class TestStagedDispatchContract(unittest.TestCase):
    """Parameter validation runs before the NPU/device guard."""

    def test_staged_mtp_above_two_rejected(self):
        topk = torch.zeros((2, 8), dtype=torch.int32)
        boundary = torch.zeros(2, dtype=torch.int32)
        table = torch.zeros((2, 4), dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "MTP=1 or MTP=2"):
            prepare_sparse_indices(
                topk, boundary, table, 128, torch.device("cpu"),
                staged_mtp=3,
            )

    def test_staged_requires_workspace(self):
        topk = torch.zeros((2, 8), dtype=torch.int32)
        boundary = torch.zeros(2, dtype=torch.int32)
        table = torch.zeros((2, 4), dtype=torch.int32)
        with self.assertRaisesRegex(
            ValueError, "local_to_union_workspace is required"
        ):
            prepare_sparse_indices(
                topk, boundary, table, 128, torch.device("cpu"),
                staged_mtp=1,
            )

    def test_staged_cpu_tensor_rejected_after_validation(self):
        topk = torch.zeros((2, 8), dtype=torch.int32)
        boundary = torch.zeros(2, dtype=torch.int32)
        table = torch.zeros((2, 4), dtype=torch.int32)
        workspace = torch.zeros((2, 8), dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "NPU custom op"):
            prepare_sparse_indices(
                topk, boundary, table, 128, torch.device("cpu"),
                local_to_union_workspace=workspace,
                staged_mtp=1,
            )


class TestStagedSingleRowSemantics(unittest.TestCase):
    """Q1 layout (staged_mtp=1: one top-k row per request) source-order
    layout vs the sorted-oracle invariants.

    staged_mtp counts top-k ROWS per request (= decode_threshold), not the
    speculative depth: 1 is the Q1 layout (MTP off / first decode step),
    2 is the MTP1 verify layout (target + draft rows).
    """

    def _case(self, seed):
        gen = torch.Generator().manual_seed(seed)
        requests = 4
        width = 32
        boundary = torch.tensor([20, 0, 40, 13], dtype=torch.int32)
        tokens = torch.randint(
            0, 60, (requests, width), generator=gen, dtype=torch.int32
        )
        # sprinkle padding entries and in-row duplicates
        tokens[0, 3] = -1
        tokens[3, 5] = tokens[3, 9]
        tokens[2, 7] = -2
        tokens[2, 8] = tokens[2, 1]
        table = torch.arange(requests * 8, dtype=torch.int32).reshape(
            requests, 8
        )
        block_size = 4
        row_req = torch.arange(requests, dtype=torch.int32)
        return tokens, boundary, table, block_size, row_req

    def test_new_indices_and_counts_match_sorted_oracle(self):
        tokens, boundary, table, block_size, row_req = self._case(7)
        ref_indices, ref_packed, ref_counts, _ = _prepare_sparse_indices_torch(
            tokens, boundary, row_req, table, block_size
        )
        (
            staged_indices,
            staged_packed,
            staged_counts,
            staged_targets,
        ) = _staged_mtp1_reference(tokens, boundary, table, block_size)

        # Hard invariants: the FA-read remap and per-request counts are
        # identical between the sorted-union and source-order layouts.
        torch.testing.assert_close(staged_indices, ref_indices)
        torch.testing.assert_close(staged_counts, ref_counts)

        # Soft invariants: the packed list is the same set of tokens (order
        # differs by design); targets follow the packed order, so the
        # (token -> physical slot) relation matches elementwise.
        for row in range(tokens.shape[0]):
            count = int(ref_counts[row])
            self.assertEqual(
                sorted(staged_packed[row, :count].tolist()),
                sorted(ref_packed[row, :count].tolist()),
            )
            for rank in range(count):
                token = int(ref_packed[row, rank])
                staged_rank = int(
                    (staged_packed[row] == token).nonzero()[0, 0]
                )
                block_id = int(table[row, staged_rank // block_size])
                self.assertEqual(
                    int(staged_targets[row, staged_rank]),
                    block_id * block_size + staged_rank % block_size,
                )

    def test_zero_boundary_selects_nothing(self):
        tokens, boundary, table, block_size, _ = self._case(11)
        boundary[:] = 0
        row_req = torch.arange(tokens.shape[0], dtype=torch.int32)
        ref_indices, _, ref_counts, _ = _prepare_sparse_indices_torch(
            tokens, boundary, row_req, table, block_size
        )
        staged_indices, _, staged_counts, _ = _staged_mtp1_reference(
            tokens, boundary, table, block_size
        )
        torch.testing.assert_close(staged_indices, tokens.long())
        torch.testing.assert_close(staged_indices, ref_indices)
        torch.testing.assert_close(staged_counts, ref_counts)
        self.assertEqual(int(ref_counts.sum()), 0)


if __name__ == "__main__":
    unittest.main()
