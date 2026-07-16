# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Any

import torch


def diagnostic_values_to_list(
    values: torch.Tensor | Iterable[Any] | None,
) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


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
