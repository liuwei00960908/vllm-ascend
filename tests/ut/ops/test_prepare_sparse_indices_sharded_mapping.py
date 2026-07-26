import random

import pytest

TOPK = 2048
INT32_BYTES = 4
CACHELINE_BYTES = 64
VECTOR_TRANSACTION_BYTES = 256


def _shard_count(mtp: int) -> int:
    return 1 << (mtp - 1).bit_length()


def _mapping_ranges(mtp: int):
    shards = _shard_count(mtp)
    request_width = mtp * TOPK
    part_width = request_width // shards
    return [(part * part_width, (part + 1) * part_width) for part in range(shards)]


def _owned_cachelines(begin: int, end: int, element_bytes: int):
    begin_bytes = begin * element_bytes
    end_bytes = end * element_bytes
    assert begin_bytes % CACHELINE_BYTES == 0
    assert end_bytes % CACHELINE_BYTES == 0
    return set(
        range(
            begin_bytes // CACHELINE_BYTES,
            end_bytes // CACHELINE_BYTES,
        )
    )


def _assert_disjoint(owners):
    for index, owned in enumerate(owners):
        for sibling in owners[index + 1 :]:
            assert owned.isdisjoint(sibling)


def _reference_sharded_union(rows, boundaries):
    shards = _shard_count(len(rows))
    shard_unions = []
    shard_local_ranks = []
    for shard in range(shards):
        if len(rows) == 1:
            tokens = [token for token in rows[0] if 0 <= token < boundaries[0]]
        else:
            tokens = sorted(
                {
                    token
                    for row, boundary in zip(rows, boundaries)
                    for token in row
                    if 0 <= token < boundary and token % shards == shard
                }
            )
        shard_unions.append(tokens)
        shard_local_ranks.append({token: rank for rank, token in enumerate(tokens)})

    offsets = []
    count = 0
    for tokens in shard_unions:
        offsets.append(count)
        count += len(tokens)
    packed = [token for shard_tokens in shard_unions for token in shard_tokens]
    remapped = []
    for row, boundary in zip(rows, boundaries):
        mapped_row = []
        for token in row:
            if 0 <= token < boundary:
                shard = token % shards
                mapped_row.append(offsets[shard] + shard_local_ranks[shard][token])
            else:
                mapped_row.append(token)
        remapped.append(mapped_row)
    return packed, remapped, shard_unions, offsets


@pytest.mark.parametrize("mtp", range(1, 9))
def test_mapping_parts_own_disjoint_cachelines_and_transactions(mtp):
    ranges = _mapping_ranges(mtp)
    request_width = mtp * TOPK
    assert ranges[0][0] == 0
    assert ranges[-1][1] == request_width
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))

    owned_cachelines = []
    owned_transactions = []
    for begin, end in ranges:
        begin_bytes = begin * INT32_BYTES
        end_bytes = end * INT32_BYTES
        assert begin_bytes % VECTOR_TRANSACTION_BYTES == 0
        assert end_bytes % VECTOR_TRANSACTION_BYTES == 0
        owned_cachelines.append(
            set(
                range(
                    begin_bytes // CACHELINE_BYTES,
                    end_bytes // CACHELINE_BYTES,
                )
            )
        )
        owned_transactions.append(
            set(
                range(
                    begin_bytes // VECTOR_TRANSACTION_BYTES,
                    end_bytes // VECTOR_TRANSACTION_BYTES,
                )
            )
        )

    for index, cachelines in enumerate(owned_cachelines):
        for sibling in owned_cachelines[index + 1 :]:
            assert cachelines.isdisjoint(sibling)
    for index, transactions in enumerate(owned_transactions):
        for sibling in owned_transactions[index + 1 :]:
            assert transactions.isdisjoint(sibling)


@pytest.mark.parametrize("mtp", range(1, 9))
def test_every_stage_and_finalize_owner_has_private_cachelines(mtp):
    requests = 3
    shards = _shard_count(mtp)
    request_width = mtp * TOPK

    # Stage has one AICore per (request, shard). Each owner gets a full token
    # row, pair row, and padded scalar-count cacheline.
    stage_owners = requests * shards
    for owner_width, element_bytes in (
        (TOPK, INT32_BYTES),
        (2 * TOPK, INT32_BYTES),
        (16, INT32_BYTES),
    ):
        ranges = [
            _owned_cachelines(
                owner * owner_width,
                (owner + 1) * owner_width,
                element_bytes,
            )
            for owner in range(stage_owners)
        ]
        _assert_disjoint(ranges)

    # Map has one AICore per (request, shard, part). Absolute offsets include
    # the enclosing shard segment, so this also checks adjacent shards.
    part_width = request_width // shards
    map_ranges = []
    for request in range(requests):
        for shard in range(shards):
            shard_base = (request * shards + shard) * request_width
            for part in range(shards):
                begin = shard_base + part * part_width
                end = begin + part_width
                assert begin * INT32_BYTES % VECTOR_TRANSACTION_BYTES == 0
                assert end * INT32_BYTES % VECTOR_TRANSACTION_BYTES == 0
                map_ranges.append(_owned_cachelines(begin, end, INT32_BYTES))
    _assert_disjoint(map_ranges)

    # Finalize has one AICore per request. All vector outputs use whole
    # request segments; selected_count uses one padded cacheline per request.
    for owner_width, element_bytes in (
        (request_width, INT32_BYTES),
        (request_width, 8),
        (16, INT32_BYTES),
    ):
        ranges = [
            _owned_cachelines(
                request * owner_width,
                (request + 1) * owner_width,
                element_bytes,
            )
            for request in range(requests)
        ]
        _assert_disjoint(ranges)


@pytest.mark.parametrize("mtp", range(1, 9))
def test_shard_local_ranks_reconstruct_without_cross_shard_order(mtp):
    rng = random.Random(20260726 + mtp)
    shared = list(range(TOPK // 2))
    rows = []
    for row_index in range(mtp):
        unique = list(
            range(
                TOPK // 2 + row_index * TOPK // 2,
                TOPK // 2 + (row_index + 1) * TOPK // 2,
            )
        )
        row = shared + unique
        rng.shuffle(row)
        rows.append(row)
    source_max = (mtp + 1) * TOPK // 2 - 1
    boundaries = [source_max - 100] * mtp

    packed, remapped, shard_unions, offsets = _reference_sharded_union(rows, boundaries)
    assert len(packed) == len(set(packed))
    assert set(packed) == set(range(source_max - 100))
    assert offsets[0] == 0
    assert all(
        offsets[shard] == sum(len(tokens) for tokens in shard_unions[:shard]) for shard in range(len(shard_unions))
    )
    for shard, tokens in enumerate(shard_unions):
        if mtp > 1:
            assert tokens == sorted(tokens)
        assert all(token % len(shard_unions) == shard for token in tokens)

    for row, mapped, boundary in zip(rows, remapped, boundaries):
        reconstructed = [packed[rank] if 0 <= token < boundary else rank for token, rank in zip(row, mapped)]
        assert reconstructed == row


@pytest.mark.parametrize("mtp", range(2, 9))
def test_union_requires_only_shard_local_order(mtp):
    rows = [list(range(row * TOPK // 2, row * TOPK // 2 + TOPK)) for row in range(mtp)]
    boundaries = [mtp * TOPK] * mtp
    packed, remapped, shard_unions, _ = _reference_sharded_union(rows, boundaries)

    assert all(tokens == sorted(tokens) for tokens in shard_unions)
    assert packed != sorted(packed)
    for row, mapped in zip(rows, remapped):
        assert [packed[rank] for rank in mapped] == row


def test_shard_completion_order_does_not_change_fixed_slot_finalize():
    mtp = 5
    rows = [list(range(row * 700, row * 700 + TOPK)) for row in range(mtp)]
    boundaries = [4200, 3900, 3600, 3300, 3000]
    packed, remapped, shard_unions, offsets = _reference_sharded_union(rows, boundaries)
    completion_order = [6, 0, 4, 7, 1, 5, 3, 2]

    scratch_by_shard = {}
    for shard in completion_order:
        scratch_by_shard[shard] = shard_unions[shard]
    finalized = [token for shard in range(_shard_count(mtp)) for token in scratch_by_shard[shard]]

    assert finalized == packed
    assert offsets == [sum(len(shard_unions[prior]) for prior in range(shard)) for shard in range(_shard_count(mtp))]
    for row, mapped, boundary in zip(rows, remapped, boundaries):
        for token, rank in zip(row, mapped):
            if token < boundary:
                assert finalized[rank] == token
            else:
                assert rank == token


def test_split_boundary_is_row_local_and_excludes_equal_or_negative_tokens():
    rows = [
        [-4, 0, 1, 2, 5, 8],
        [-1, 1, 3, 5, 7, 9],
        [0, 2, 4, 6, 8, 10],
    ]
    boundaries = [5, 7, 9]
    packed, remapped, _, _ = _reference_sharded_union(rows, boundaries)

    assert set(packed) == {0, 1, 2, 3, 4, 5, 6, 8}
    for row, mapped, boundary in zip(rows, remapped, boundaries):
        for token, rank in zip(row, mapped):
            if 0 <= token < boundary:
                assert packed[rank] == token
            else:
                assert rank == token


def test_mtp1_fast_path_preserves_compacted_input_order():
    row = [9, 1, 7, 3, 5, 0, 8, 2, 6, 4]
    boundary = 7
    packed = [token for token in row if 0 <= token < boundary]
    mapping = {}
    for rank, token in enumerate(packed):
        mapping[token] = rank

    assert packed == [1, 3, 5, 0, 2, 6, 4]
    assert len(packed) == len(set(packed))
    assert [mapping[token] if 0 <= token < boundary else token for token in row] == [9, 0, 7, 1, 2, 3, 8, 4, 5, 6]
