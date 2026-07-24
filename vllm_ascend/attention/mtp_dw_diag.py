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


def scheduled_decode_requests(
    request_ids: Iterable[Any],
    scheduled_request_ids: Iterable[str],
    num_computed_tokens: Iterable[int],
    num_prompt_tokens: Iterable[int],
) -> list[tuple[int, str]]:
    """Return input-batch rows that are scheduled for decode in this step."""
    scheduled = set(scheduled_request_ids)
    computed = list(num_computed_tokens)
    prompt = list(num_prompt_tokens)
    result: list[tuple[int, str]] = []
    for request_index, raw_request_id in enumerate(request_ids):
        request_id = str(raw_request_id)
        if request_id not in scheduled:
            continue
        if int(computed[request_index]) < int(prompt[request_index]):
            continue
        result.append((request_index, request_id))
    return result


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


def scratch_target_safety(
    block_table: Iterable[int],
    scratch_start: int,
    scratch_count: int,
    committed_end: int,
    current_position: int,
    block_size: int,
) -> dict[str, Any]:
    """Describe whether an active scratch range is safe to overwrite.

    Scratch target positions are resolved through the normal request block
    table. This diagnostic is intentionally read-only: it reports the logical
    range, block-table coverage, and physical overlap without changing remap
    or transfer behavior.
    """
    if scratch_start < 0 or scratch_count < 0:
        raise ValueError("scratch range must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    blocks = [int(block) for block in block_table]
    target_end = scratch_start + scratch_count
    valid_block_count = next(
        (index for index, block in enumerate(blocks) if block < 0), len(blocks)
    )
    valid_logical_end = valid_block_count * block_size
    target_block_start = scratch_start // block_size
    target_block_end = (target_end + block_size - 1) // block_size
    target_block_values = [
        blocks[index] if index < len(blocks) else None
        for index in range(target_block_start, target_block_end)
    ]
    mapped_end = min(target_end, valid_logical_end)
    mapped_count = max(0, mapped_end - scratch_start)
    mapped_positions = range(scratch_start, max(scratch_start, mapped_end))
    target_slots = logical_to_physical_slots(blocks, mapped_positions, block_size)
    live_start = min(max(committed_end, 0), valid_logical_end)
    live_end = min(max(current_position + 1, live_start), valid_logical_end)
    live_slots = logical_to_physical_slots(
        blocks, range(live_start, live_end), block_size
    )

    return {
        "target_logical_start": scratch_start,
        "target_logical_end": target_end,
        "target_block_start": target_block_start,
        "target_block_end": target_block_end,
        "target_block_values": target_block_values,
        "valid_logical_end": valid_logical_end,
        "target_within_committed": target_end <= committed_end,
        "target_beyond_current_sequence": target_end > current_position + 1,
        "target_unmapped_count": scratch_count - mapped_count,
        "target_slots": target_slots,
        "live_slots": live_slots,
        "target_live_intersection": sorted(
            set(target_slots).intersection(live_slots)
        ),
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
