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
# Unit tests for DSA replay env registration and composite gates
# (replay Step 1 / A1). These tests only exercise vllm_ascend.envs
# lazy evaluation; no NPU or torch import is required.
#

import os
from contextlib import contextmanager

import pytest

from vllm_ascend import envs

_NAMES = (
    "VLLM_ASCEND_DSA_UNBUNDLE",
    "VLLM_ASCEND_DSA_TWO_GROUPS",
    "VLLM_ASCEND_DSA_SHARED_POOL",
    "VLLM_ASCEND_DSA_SHRINK_LATENT",
)


@contextmanager
def _dsa_env(**kwargs):
    saved = {name: os.environ.get(name) for name in _NAMES}
    try:
        for name in _NAMES:
            value = kwargs.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def test_defaults():
    with _dsa_env():
        assert envs.VLLM_ASCEND_DSA_UNBUNDLE is False
        assert envs.VLLM_ASCEND_DSA_TWO_GROUPS is False
        # Raw default "1" mirrors the fork; suppression happens in the
        # composite gate (model_runner), not in the raw env value.
        assert envs.VLLM_ASCEND_DSA_SHARED_POOL is True
        assert envs.VLLM_ASCEND_DSA_SHRINK_LATENT == 0


def test_lazy_eval_reflects_env_changes():
    with _dsa_env(VLLM_ASCEND_DSA_UNBUNDLE="1", VLLM_ASCEND_DSA_SHRINK_LATENT="2"):
        assert envs.VLLM_ASCEND_DSA_UNBUNDLE is True
        assert envs.VLLM_ASCEND_DSA_SHRINK_LATENT == 2
        assert envs.VLLM_ASCEND_DSA_TWO_GROUPS is False
    with _dsa_env():
        assert envs.VLLM_ASCEND_DSA_UNBUNDLE is False


def test_shrink_empty_string_falls_back_to_zero():
    with _dsa_env(VLLM_ASCEND_DSA_SHRINK_LATENT=""):
        assert envs.VLLM_ASCEND_DSA_SHRINK_LATENT == 0


def test_invalid_shrink_raises():
    # Non-integer values must fail loudly rather than silently defaulting.
    with _dsa_env(VLLM_ASCEND_DSA_SHRINK_LATENT="abc"):
        with pytest.raises(ValueError):
            _ = envs.VLLM_ASCEND_DSA_SHRINK_LATENT


if __name__ == "__main__":
    pytest.main([__file__])
