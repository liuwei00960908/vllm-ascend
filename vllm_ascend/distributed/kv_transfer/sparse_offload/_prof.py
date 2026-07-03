# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op hooks for the DSA latent-offload profiling call sites."""


def set_step_kind(is_decode_only: bool) -> None:
    return None


class section:
    __slots__ = ()

    def __init__(self, name: str) -> None:
        return None

    def __enter__(self) -> "section":
        return self

    def __exit__(self, *exc) -> None:
        return None


def begin(name: str):
    return None


def end(token) -> None:
    return None


def log_topk_padding(topk_row, invalid: int) -> None:
    return None


def step() -> None:
    return None
