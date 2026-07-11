# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profiling hooks for the DSA latent-offload call sites.

Default behaviour is full no-op (zero overhead).  Set ``VLLM_ASCEND_DSA_PROF=1``
to activate per-section host-side timing with a ``torch.npu.synchronize()`` at
each decode-step boundary so that the reported *total* reflects real device
time.

Environment variables
---------------------
VLLM_ASCEND_DSA_PROF
    ``1`` to enable.
VLLM_ASCEND_DSA_PROF_LAYERS
    Number of attention layers per model forward (default ``61``).  Used to
    detect decode-step boundaries for the aggregated summary.
VLLM_ASCEND_DSA_PROF_LIMIT
    How many per-layer lines to print before suppressing (default ``4``).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Optional

_PROF_ENABLED: bool = os.getenv("VLLM_ASCEND_DSA_PROF", "0") == "1"
_PROF_LAYERS: int = int(os.getenv("VLLM_ASCEND_DSA_PROF_LAYERS", "61"))
_PROF_LIMIT: int = int(os.getenv("VLLM_ASCEND_DSA_PROF_LIMIT", "4"))

if _PROF_ENABLED:
    _section_acc: Dict[str, float] = {}
    _layer_count: int = 0
    _printed: int = 0
    _step_start: float = 0.0
    _step_kind: str = "?"


class _NoopSection:
    __slots__ = ()

    def __enter__(self) -> "_NoopSection":
        return self

    def __exit__(self, *exc) -> None:
        return None


_NOOP_SECTION = _NoopSection()


@contextmanager
def _real_section(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        _section_acc[name] = _section_acc.get(name, 0.0) + dt


def set_step_kind(is_decode_only: bool) -> None:
    if not _PROF_ENABLED:
        return
    global _step_kind, _layer_count, _printed, _section_acc, _step_start
    _step_kind = "decode" if is_decode_only else "mixed"
    if _layer_count == 0:
        _step_start = time.perf_counter()
        _section_acc = {}
        _printed = 0


def section(name: str):
    if _PROF_ENABLED:
        return _real_section(name)
    return _NOOP_SECTION


def begin(name: str) -> Optional[tuple]:
    if not _PROF_ENABLED:
        return None
    return (name, time.perf_counter())


def end(token) -> None:
    if not _PROF_ENABLED or token is None:
        return
    name, t0 = token
    dt = (time.perf_counter() - t0) * 1000.0
    _section_acc[name] = _section_acc.get(name, 0.0) + dt


def log_topk_padding(topk_row, invalid: int) -> None:
    return None


def step() -> None:
    if not _PROF_ENABLED:
        return
    global _layer_count, _printed, _step_start
    _layer_count += 1
    if _printed < _PROF_LIMIT:
        parts = [f"{k}={v:.2f}ms" for k, v in _section_acc.items()]
        print(
            f"[DSA_PROF] layer={_layer_count - 1} kind={_step_kind} "
            + " ".join(parts),
            flush=True,
        )
        _printed += 1
    if _layer_count >= _PROF_LAYERS:
        try:
            import torch

            if hasattr(torch, "npu"):
                torch.npu.synchronize()
        except Exception:
            pass
        total = (time.perf_counter() - _step_start) * 1000.0 if _step_start else 0.0
        host_sum = sum(_section_acc.values())
        gap = total - host_sum
        parts = [f"{k}={v:.2f}ms" for k, v in _section_acc.items()]
        print(
            f"[DSA_PROF] STEP_DONE kind={_step_kind} total={total:.1f}ms "
            f"host_sum={host_sum:.1f}ms device_gap={gap:.1f}ms "
            f"layers={_layer_count} "
            + " ".join(parts),
            flush=True,
        )
        _layer_count = 0
