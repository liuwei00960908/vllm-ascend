# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm_ascend.attention.mtp_dw_diag import (
    diagnostic_int_checksum,
    diagnostic_values_to_list,
    logical_to_physical_slots,
    post_commit_sample_requests,
    scratch_live_slot_aliases,
    scratch_target_safety,
)


def test_diagnostic_values_to_list_handles_none_tensor_and_numpy() -> None:
    assert diagnostic_values_to_list(None) == []
    assert diagnostic_values_to_list(torch.tensor([2, 0, 1])) == [2, 0, 1]
    assert diagnostic_values_to_list(np.array([1, 3, 2])) == [1, 3, 2]


def test_post_commit_sampling_forces_first_nonzero_and_changes() -> None:
    previous: dict[str, int] = {}

    request_ids = np.array(["req"])
    assert post_commit_sample_requests(previous, request_ids, np.array([0])) == set()
    assert post_commit_sample_requests(
        previous, request_ids, np.array([256])
    ) == {"req"}
    assert post_commit_sample_requests(previous, ["req"], [256]) == set()
    assert post_commit_sample_requests(previous, ["req"], [512]) == {"req"}


def test_diagnostic_int_checksum_is_stable_ordered_and_bounded() -> None:
    values = [12, -1, 999_999_999_999]
    checksum = diagnostic_int_checksum(values)

    assert checksum == diagnostic_int_checksum(tuple(values))
    assert checksum != diagnostic_int_checksum(reversed(values))
    assert 0 <= checksum <= (1 << 64) - 1
    assert diagnostic_int_checksum(values + list(range(40))) == diagnostic_int_checksum(
        (values + list(range(40)))[:32] + [999] * 8
    )
    assert diagnostic_int_checksum([]) == 0xCBF29CE484222325


def test_scratch_live_slot_aliases_use_physical_block_mapping() -> None:
    block_table = [7, 3, 7, 9]

    assert logical_to_physical_slots(block_table, [0, 5, 8], 4) == [28, 13, 28]
    target, live, aliases = scratch_live_slot_aliases(
        block_table,
        scratch_positions=[0, 1],
        live_start=8,
        current_position=9,
        block_size=4,
    )
    assert target == [28, 29]
    assert live == [28, 29]
    assert aliases == [28, 29]


def test_scratch_live_slot_aliases_reports_disjoint_ranges() -> None:
    _, _, aliases = scratch_live_slot_aliases([7, 3, 5], [0, 1], 8, 9, 4)
    assert aliases == []


def test_scratch_alias_check_excludes_intentionally_reused_offloaded_region() -> None:
    target, live, aliases = scratch_live_slot_aliases(
        [7, 3, 7, 9], [0, 1], live_start=12, current_position=13, block_size=4
    )

    assert target == [28, 29]
    assert live == [36, 37]
    assert aliases == []


def test_scratch_alias_check_ignores_unallocated_block_table_tail() -> None:
    target, live, aliases = scratch_live_slot_aliases(
        [7, 3, -1, -1],
        [0, 1],
        live_start=6,
        current_position=31,
        block_size=4,
    )

    assert target == [28, 29]
    assert live == [14, 15]
    assert aliases == []


def test_scratch_alias_check_accepts_one_shot_block_iterables() -> None:
    target, live, aliases = scratch_live_slot_aliases(
        iter([7, 3, 5]), [0], live_start=8, current_position=8, block_size=4
    )

    assert target == [28]
    assert live == [20]
    assert aliases == []


def test_scratch_target_safety_reports_live_overlap() -> None:
    safety = scratch_target_safety(
        [7, 3, 5, 9],
        scratch_start=8,
        scratch_count=2,
        committed_end=2,
        current_position=9,
        block_size=4,
    )

    assert safety["target_logical_start"] == 8
    assert safety["target_logical_end"] == 10
    assert safety["target_block_values"] == [5]
    assert safety["target_within_committed"] is False
    assert safety["target_beyond_current_sequence"] is False
    assert safety["target_live_intersection"] == [20, 21]


def test_scratch_target_safety_reports_unmapped_tail() -> None:
    safety = scratch_target_safety(
        [7, 3, -1, -1],
        scratch_start=8,
        scratch_count=2,
        committed_end=2,
        current_position=9,
        block_size=4,
    )

    assert safety["target_block_values"] == [-1]
    assert safety["target_unmapped_count"] == 2
    assert safety["target_live_intersection"] == []


def test_scratch_target_safety_detects_out_of_range_blocks() -> None:
    """Stale residual block-table entries past num_blocks must be flagged."""
    safety = scratch_target_safety(
        [10, 20, 30, 40],
        scratch_start=8,
        scratch_count=4,
        committed_end=4,
        current_position=9,
        block_size=4,
        num_blocks=2,
    )

    assert safety["num_blocks"] == 2
    assert safety["target_block_start"] == 2
    assert safety["target_block_end"] == 3
    assert safety["target_blocks_out_of_range"] == 2
    assert safety["target_tokens_out_of_range"] == 4


def test_scratch_target_safety_all_blocks_in_range() -> None:
    """When all scratch targets fall within allocated blocks, oor is 0."""
    safety = scratch_target_safety(
        [10, 20, 30, 40],
        scratch_start=0,
        scratch_count=8,
        committed_end=8,
        current_position=9,
        block_size=4,
        num_blocks=4,
    )

    assert safety["target_blocks_out_of_range"] == 0
    assert safety["target_tokens_out_of_range"] == 0


def test_scratch_target_safety_num_blocks_none_disables_oor() -> None:
    """Without num_blocks the oor fields should be None."""
    safety = scratch_target_safety(
        [10, 20],
        scratch_start=0,
        scratch_count=4,
        committed_end=4,
        current_position=5,
        block_size=4,
    )

    assert safety["num_blocks"] is None
    assert safety["target_blocks_out_of_range"] is None
    assert safety["target_tokens_out_of_range"] is None
