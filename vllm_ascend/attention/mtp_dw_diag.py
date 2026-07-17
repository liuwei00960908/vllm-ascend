# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch

_CHECKSUM_LIMIT = 32
_UINT64_MASK = (1 << 64) - 1


def diagnostic_values_to_list(
    values: torch.Tensor | Iterable[Any] | None,
) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


def diagnostic_int_checksum(values: Iterable[int]) -> int:
    """Match LMCache-Ascend's stable checksum over the first 32 integers."""
    prefix = list(islice(values, _CHECKSUM_LIMIT))
    checksum = 0xCBF29CE484222325
    for value in prefix:
        checksum ^= int(value) & _UINT64_MASK
        checksum = (checksum * 0x100000001B3) & _UINT64_MASK
    checksum ^= len(prefix)
    return checksum


def logical_to_physical_slots(
    block_table: Iterable[int], logical_positions: Iterable[int], block_size: int
) -> list[int]:
    """Resolve logical token positions through one CPU block-table row."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks = [int(block) for block in block_table]
    slots: list[int] = []
    for position in logical_positions:
        position = int(position)
        if position < 0:
            raise ValueError("logical positions must be non-negative")
        logical_block, offset = divmod(position, block_size)
        if logical_block >= len(blocks):
            raise ValueError("logical position exceeds block table")
        if blocks[logical_block] < 0:
            raise ValueError("logical position exceeds valid block table")
        slots.append(blocks[logical_block] * block_size + offset)
    return slots


def scratch_live_slot_aliases(
    block_table: Iterable[int],
    scratch_positions: Iterable[int],
    live_start: int,
    current_position: int,
    block_size: int,
) -> tuple[list[int], list[int], list[int]]:
    """Resolve scratch/live slots and return their sorted physical intersection."""
    blocks = [int(block) for block in block_table]
    target_slots = logical_to_physical_slots(
        blocks, scratch_positions, block_size
    )
    valid_blocks = next(
        (index for index, block in enumerate(blocks) if block < 0), len(blocks)
    )
    live_end = min(current_position + 1, valid_blocks * block_size)
    live_positions = range(live_start, max(live_start, live_end))
    live_slots = logical_to_physical_slots(blocks, live_positions, block_size)
    aliases = sorted(set(target_slots).intersection(live_slots))
    return target_slots, live_slots, aliases


def first_post_commit_requests(
    previous_frontiers: dict[str, int],
    request_ids: Iterable[str],
    committed_frontiers: Iterable[int],
) -> set[str]:
    """Return requests making their first transition to a committed prefix."""
    return {
        str(req_id)
        for req_id, committed in zip(request_ids, committed_frontiers)
        if int(committed) > 0
        and previous_frontiers.get(str(req_id), 0) <= 0
    }


def post_commit_sample_requests(
    previous_frontiers: dict[str, int],
    request_ids: Iterable[str],
    committed_frontiers: Iterable[int],
) -> set[str]:
    """Return requests whose committed frontier first became readable or changed."""
    sampled: set[str] = set()
    for req_id, committed in zip(request_ids, committed_frontiers):
        req_id = str(req_id)
        committed = int(committed)
        previous = previous_frontiers.get(req_id, 0)
        if committed > 0 and committed != previous:
            sampled.add(req_id)
        previous_frontiers[req_id] = committed
    return sampled
