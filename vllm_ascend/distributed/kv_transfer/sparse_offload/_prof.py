# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated micro-profiler for the DSA latent-offload decode path.

Enable with ``VLLM_ASCEND_DSA_OFFLOAD_PROFILE=1``. Each :class:`section` brackets a
piece of the path with ``torch.npu.synchronize()`` so the measured time is real device
time (not just launch time), accumulates mean ms/call, and logs a summary every
``_LOG_EVERY`` calls. Zero cost when disabled (no synchronize, no timing).
"""

import os
import time
from collections import defaultdict

import torch
from vllm.logger import logger

ENABLED = bool(int(os.getenv("VLLM_ASCEND_DSA_OFFLOAD_PROFILE", "0")))

_acc: dict[str, float] = defaultdict(float)
_n: dict[str, int] = defaultdict(int)
_calls = [0]
_LOG_EVERY = 160  # ~20 decode tokens at 8 layers each


class section:
    """Context manager timing a named device section (no-op unless ENABLED)."""

    __slots__ = ("name", "_t")

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "section":
        if ENABLED:
            torch.npu.synchronize()
            self._t = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if ENABLED:
            torch.npu.synchronize()
            _acc[self.name] += (time.perf_counter() - self._t) * 1000.0
            _n[self.name] += 1


def step() -> None:
    """Call once per decode layer; periodically logs mean ms/call per section."""
    if not ENABLED:
        return
    _calls[0] += 1
    if _calls[0] % _LOG_EVERY == 0:
        parts = "  ".join(
            f"{k}={_acc[k] / _n[k]:.3f}" for k in sorted(_acc)
        )
        total = sum(_acc[k] / _n[k] for k in _acc)
        logger.info(
            "[DSA-PROF] mean ms/layer-call over %d calls: %s  (sum=%.3f)",
            _calls[0], parts, total,
        )
