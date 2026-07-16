# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm_ascend.attention.mtp_dw_diag import diagnostic_values_to_list


def test_diagnostic_values_to_list_handles_none_tensor_and_numpy() -> None:
    assert diagnostic_values_to_list(None) == []
    assert diagnostic_values_to_list(torch.tensor([2, 0, 1])) == [2, 0, 1]
    assert diagnostic_values_to_list(np.array([1, 3, 2])) == [1, 3, 2]
