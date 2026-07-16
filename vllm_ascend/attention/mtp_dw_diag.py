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
