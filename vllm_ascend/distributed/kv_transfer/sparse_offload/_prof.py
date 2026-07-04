# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op hooks for the DSA latent-offload profiling call sites."""


def set_step_kind(is_decode_only: bool) -> None:
    return None


class _NoopSection:
    __slots__ = ()

    def __enter__(self) -> "_NoopSection":
        return self

    def __exit__(self, *exc) -> None:
        return None


_NOOP_SECTION = _NoopSection()


def section(name: str) -> _NoopSection:
    return _NOOP_SECTION


def begin(name: str):
    return None


def end(token) -> None:
    return None


def log_topk_padding(topk_row, invalid: int) -> None:
    return None


def step() -> None:
    return None
