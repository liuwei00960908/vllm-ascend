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


def begin(name: str):
    """Manual span start for code with early returns (pairs with end()). Returns a
    token to pass to end(), or None when disabled."""
    if not ENABLED:
        return None
    torch.npu.synchronize()
    return (name, time.perf_counter())


def end(token) -> None:
    """Manual span stop; accumulates real device time into the named section."""
    if not ENABLED or token is None:
        return
    torch.npu.synchronize()
    name, t = token
    _acc[name] += (time.perf_counter() - t) * 1000.0
    _n[name] += 1


_padding_logged = [False]


def log_topk_padding(topk_row: "torch.Tensor", invalid: int) -> None:
    """One-shot: report whether a topk row is front-packed (valid first, -1 padding at
    the end) — which would let us drop the compaction in build_gather_plan."""
    if not ENABLED or _padding_logged[0]:
        return
    _padding_logged[0] = True
    valid = topk_row != invalid
    n_valid = int(valid.sum())
    inv = (~valid).nonzero()
    first_invalid = int(inv[0]) if inv.numel() else topk_row.numel()
    logger.info(
        "[DSA-PROF] topk padding: len=%d n_valid=%d first_invalid_idx=%d front_packed=%s",
        topk_row.numel(), n_valid, first_invalid, n_valid == first_invalid,
    )


def step() -> None:
    """Call once per decode layer; periodically logs mean ms/call per section.

    WINDOWED: stats are reset after each log so every line reflects only the last
    _LOG_EVERY layer-calls. This keeps a few heavy prefill calls (16k tokens each)
    from dominating the cumulative mean — once in steady decode, the latest lines
    are pure-decode timings.
    """
    if not ENABLED:
        return
    _calls[0] += 1
    if _calls[0] % _LOG_EVERY == 0:
        parts = "  ".join(
            f"{k}={_acc[k] / _n[k]:.3f}" for k in sorted(_acc) if _n[k]
        )
        total = sum(_acc[k] / _n[k] for k in _acc if _n[k])
        logger.info(
            "[DSA-PROF] window mean ms/layer-call (last %d): %s  (sum=%.3f)",
            _LOG_EVERY, parts, total,
        )
        _acc.clear()
        _n.clear()
