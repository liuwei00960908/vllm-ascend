#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Profile and summarize multi-request sparse KV copy kernel execution."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tools import staged_sfa_graph_smoke as smoke

_NEW_HOST_OP = "sparse_mla_dsa_multi_request_direct_kv_transfer_prepared"
_NEW_DEVICE_MARKER = "_sparse_multi_request_"
_LEGACY_DEVICE_MARKER = "_sparse_multi_chunk_"
_DEVICE_PREFIX = "single_layer_paged_kv_copy_v2_"
_RETRIEVE_SCOPE = "sfa_cross_layer::lmcache_retrieve"
_LOCAL_LAYER_PATTERN = re.compile(
    re.escape(smoke._STARTUP_CROSS_LAYER_COMPLETE) + r" for (\d+) local SFA layers and (\d+) keys"
)
_ERROR_MARKERS = (
    "507057",
    "DDR address out of range",
    "selected_oob=True",
    "slot_selected_match=False",
    "EngineDeadError",
    "ascend_model_forward_exception",
)


class VerificationFailure(RuntimeError):
    """The sparse batch verification could not establish its trace contract."""


def _clear_profile_directory(profile_dir: Path) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    for child in profile_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _trace_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise VerificationFailure(f"unexpected trace JSON shape: {path}")
    return [event for event in events if isinstance(event, dict)]


def _count_trace(path: Path) -> dict[str, Any]:
    names = Counter(str(event.get("name", "")) for event in _trace_events(path))
    new_device_names = sorted(name for name in names if name.startswith(_DEVICE_PREFIX) and _NEW_DEVICE_MARKER in name)
    legacy_device_names = sorted(
        name for name in names if name.startswith(_DEVICE_PREFIX) and _LEGACY_DEVICE_MARKER in name
    )
    return {
        "trace": str(path),
        "new_host_op_count": names[_NEW_HOST_OP],
        "new_device_kernel_count": sum(names[name] for name in new_device_names),
        "legacy_device_kernel_count": sum(names[name] for name in legacy_device_names),
        "lmcache_retrieve_scope_count": names[_RETRIEVE_SCOPE],
        "new_device_kernel_names": new_device_names,
        "legacy_device_kernel_names": legacy_device_names,
    }


def _server_metadata(server_log: Path) -> tuple[int | None, int | None]:
    text = server_log.read_text(encoding="utf-8", errors="replace")
    matches = list(_LOCAL_LAYER_PATTERN.finditer(text))
    if not matches:
        return None, None
    match = matches[-1]
    return int(match.group(1)), int(match.group(2))


def _server_errors(server_log: Path, start_offset: int) -> dict[str, int]:
    with server_log.open("rb") as log_file:
        log_file.seek(start_offset)
        text = log_file.read().decode("utf-8", errors="replace")
    return {marker: text.count(marker) for marker in _ERROR_MARKERS}


def summarize(
    profile_dir: Path,
    server_log: Path,
    *,
    batch_size: int,
    log_start_offset: int,
) -> dict[str, Any]:
    """Summarize sparse copy kernels emitted by all newly written worker traces."""
    local_layers, graph_keys = _server_metadata(server_log)
    traces = sorted(profile_dir.rglob("trace_view.json"))
    rank_summaries = []
    for trace in traces:
        try:
            rank_summaries.append(_count_trace(trace))
        except (OSError, json.JSONDecodeError, VerificationFailure) as error:
            rank_summaries.append({"trace": str(trace), "error": str(error)})

    for rank in rank_summaries:
        if local_layers and "new_device_kernel_count" in rank:
            rank["new_device_per_local_layer"] = rank["new_device_kernel_count"] / local_layers
            rank["new_host_per_local_layer"] = rank["new_host_op_count"] / local_layers

    aggregate = Counter()
    for rank in rank_summaries:
        for key in (
            "new_host_op_count",
            "new_device_kernel_count",
            "legacy_device_kernel_count",
            "lmcache_retrieve_scope_count",
        ):
            aggregate[key] += int(rank.get(key, 0))

    errors = _server_errors(server_log, log_start_offset)
    failures = []
    warnings = []
    valid_ranks = [rank for rank in rank_summaries if "error" not in rank]
    if not valid_ranks:
        failures.append("no readable trace_view.json files were produced")
    if aggregate["new_device_kernel_count"] == 0:
        failures.append("no multi-request sparse copy device kernel was recorded")
    if local_layers is None:
        warnings.append("could not determine local SFA layer count from server log")
    else:
        for rank in valid_ranks:
            count = int(rank["new_device_kernel_count"])
            if count and count % local_layers:
                warnings.append(
                    f"{rank['trace']}: new device kernel count {count} is not "
                    f"divisible by local layer count {local_layers}"
                )
    if aggregate["legacy_device_kernel_count"]:
        warnings.append(
            "legacy sparse multi-chunk kernels were recorded; verify that the "
            "profile interval started after cold/bootstrap fallback completed"
        )
    if any(errors.values()):
        failures.append("server log contains sparse retrieve failure markers")

    return {
        "requested_batch_size": batch_size,
        "local_sfa_layers": local_layers,
        "captured_graph_keys": graph_keys,
        "trace_count": len(traces),
        "rank_summaries": rank_summaries,
        "aggregate": dict(aggregate),
        "server_errors": errors,
        "failures": failures,
        "warnings": warnings,
        "result": "FAIL" if failures else "WARN" if warnings else "PASS",
    }


def _render_summary(summary: dict[str, Any]) -> str:
    lines = ["=== Sparse Batch Verification Summary ==="]
    for key in (
        "result",
        "requested_batch_size",
        "local_sfa_layers",
        "captured_graph_keys",
        "trace_count",
    ):
        lines.append(f"{key}={summary[key]}")
    lines.append("--- Aggregate ---")
    for key, value in summary["aggregate"].items():
        lines.append(f"{key}={value}")
    lines.append("--- Per trace ---")
    for rank in summary["rank_summaries"]:
        lines.append(f"trace={rank['trace']}")
        for key, value in rank.items():
            if key != "trace":
                lines.append(f"  {key}={value}")
    lines.append("--- Server errors after profile start ---")
    for key, value in summary["server_errors"].items():
        lines.append(f"{key}={value}")
    if summary["warnings"]:
        lines.append("--- Warnings ---")
        lines.extend(summary["warnings"])
    if summary["failures"]:
        lines.append("--- Failures ---")
        lines.extend(summary["failures"])
    return "\n".join(lines) + "\n"


def _smoke_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("staged_sfa_graph_smoke.py")),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--server-log",
        str(args.server_log),
        "--profile-dir",
        str(args.profile_dir),
        "--expected-ranks",
        str(args.expected_ranks),
        "--expected-keys",
        str(args.expected_keys),
        "--concurrency",
        str(args.concurrency),
        "--prompt-words",
        str(args.prompt_words),
        "--max-tokens",
        str(args.max_tokens),
        "--profile-after-chunks",
        str(args.profile_after_chunks),
        "--profile-chunks",
        str(args.profile_chunks),
        "--ready-timeout",
        str(args.ready_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--profile-control-timeout",
        str(args.profile_control_timeout),
        "--profile-analysis-timeout",
        str(args.profile_analysis_timeout),
        "--trace-timeout",
        str(args.trace_timeout),
    ]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="GLM-5.1")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--expected-keys", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt-words", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--profile-after-chunks", type=int, default=384)
    parser.add_argument("--profile-chunks", type=int, default=8)
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--profile-control-timeout", type=float, default=300)
    parser.add_argument("--profile-analysis-timeout", type=float, default=900)
    parser.add_argument("--trace-timeout", type=float, default=600)
    parser.add_argument(
        "--clean-profile-dir",
        action="store_true",
        help="delete existing profiler output before starting the client workload",
    )
    args = parser.parse_args()
    if args.concurrency <= 0 or args.expected_ranks <= 0 or args.expected_keys <= 0:
        parser.error("concurrency, expected-ranks, and expected-keys must be positive")
    if args.max_tokens < args.profile_after_chunks + args.profile_chunks + 2:
        parser.error("max-tokens is too short for the requested profile interval")
    if args.output_dir is None:
        args.output_dir = args.profile_dir / "sparse_batch_verify"
    return args


def main() -> int:
    args = parse_args()
    if not args.server_log.is_file():
        raise VerificationFailure(f"server log does not exist: {args.server_log}")
    if args.clean_profile_dir:
        _clear_profile_directory(args.profile_dir)
    elif any(args.profile_dir.rglob("trace_view.json")):
        raise VerificationFailure(
            "profile directory already contains trace_view.json; use a unique directory or pass --clean-profile-dir"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_start_offset = args.server_log.stat().st_size
    command = _smoke_command(args)
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    smoke_output = completed.stdout + completed.stderr
    (args.output_dir / "smoke.out").write_text(smoke_output, encoding="utf-8")
    print(smoke_output, end="")

    summary = summarize(
        args.profile_dir,
        args.server_log,
        batch_size=args.concurrency,
        log_start_offset=log_start_offset,
    )
    if completed.returncode:
        summary["failures"].append(f"staged_sfa_graph_smoke.py exited with status {completed.returncode}")
        summary["result"] = "FAIL"
    rendered = _render_summary(summary)
    (args.output_dir / "sparse_batch_summary.txt").write_text(rendered, encoding="utf-8")
    (args.output_dir / "sparse_batch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "server_error_scan.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in summary["server_errors"].items()) + "\n",
        encoding="utf-8",
    )
    print(rendered, end="")
    print(f"diagnostics={args.output_dir}")
    return 0 if summary["result"] != "FAIL" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationFailure as error:
        print(f"SPARSE BATCH VERIFY FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
