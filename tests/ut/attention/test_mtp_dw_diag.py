# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm_ascend.attention.mtp_dw_diag import (
    diagnostic_values_to_list,
    post_commit_sample_requests,
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
