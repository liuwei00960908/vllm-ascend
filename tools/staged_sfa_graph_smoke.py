#!/usr/bin/env python3
"""Drive and check the exact-Q1 cross-layer SFA graph milestone.

The server must already be running with the requested TP size, no speculative decoding,
SHRINK_LATENT=2, PIECEWISE graph mode, and the staged-SFA POC enabled.  This
The default sends one long streaming completion and profiles steady-state
decode. ``--concurrency N`` drives synchronized exact-Q1 batch requests; only
the first request controls the shared profiler interval.

Automated checks require successful cross-layer startup capture and worker
traces containing eager LMCache retrieval plus ACL model replay. They do not
prove output parity, timeline nesting, or device execution density; run the
deterministic output/TPOT comparison and finish the MindStudio checklist.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, BrokenBarrierError

_TRACE_MARKERS = (
    "sfa_cross_layer::bootstrap",
    "sfa_cross_layer::lmcache_retrieve",
)
_LEGACY_GRAPH_MARKERS = (
    "sfa_staged_graph_poc::pre",
    "sfa_staged_graph_poc::post",
)
_ACL_REPLAY_APIS = (
    "aclmdlRIExecuteAsync",
    "aclmdlExecuteAsync",
    "aclmdlExecuteAsyncV2",
)
_STARTUP_CROSS_LAYER_COMPLETE = "[SFA cross-layer graph] captured retrieve-split outer graphs"
_FRONTEND_PROFILER_ENABLED = "Torch profiler enabled. AsyncLLM CPU traces will be collected under"
_FAILURE_MARKERS = (
    "[SFA cross-layer graph] no local SFA layers were captured",
    "[SFA cross-layer graph] eager warmup/capture was incomplete",
    "[SFA cross-layer graph] runner-authorized key became ineligible",
    "[SFA_ROUTE] action=fatal",
)


class SmokeFailure(RuntimeError):
    """A deterministic smoke gate did not pass."""


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float,
):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    return urllib.request.urlopen(request, timeout=timeout)


def wait_until_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with _request(
                _url(base_url, "/health"),
                timeout=min(5.0, timeout),
            ) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise SmokeFailure(f"server did not become healthy within {timeout:g}s: {last_error}")


def require_worker_only_profiling(server_log: Path) -> None:
    """Reject frontend profiling before it can block the stop endpoint."""
    if not server_log.is_file():
        raise SmokeFailure(f"server log does not exist: {server_log}")
    with server_log.open("r", encoding="utf-8", errors="replace") as log_file:
        if any(_FRONTEND_PROFILER_ENABLED in line for line in log_file):
            raise SmokeFailure(
                "AsyncLLM frontend profiling is enabled; restart the server "
                "with profiler config ignore_frontend=true so /stop_profile "
                "only finalizes the TP worker traces"
            )


def profile_control(base_url: str, action: str, timeout: float) -> None:
    with _request(
        _url(base_url, f"/{action}_profile"),
        method="POST",
        timeout=timeout,
    ) as response:
        body = response.read().decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise SmokeFailure(f"{action}_profile returned HTTP {response.status}: {body}")


def analyse_profile_data(
    profile_dir: Path,
    expected_ranks: int,
    timeout: float,
) -> None:
    """Parse all raw rank traces from a non-daemon helper process."""
    print(
        f"offline parsing profiler data under {profile_dir} (timeout {timeout:g}s)...",
        flush=True,
    )
    command = [
        sys.executable,
        "-c",
        (
            "from torch_npu.profiler.profiler import analyse; "
            "import sys; analyse(sys.argv[1], "
            "max_process_number=int(sys.argv[2]))"
        ),
        str(profile_dir),
        str(expected_ranks),
    ]
    try:
        subprocess.run(command, check=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SmokeFailure(f"offline profiler analysis did not finish within {timeout:g}s: {profile_dir}") from exc
    except subprocess.CalledProcessError as exc:
        raise SmokeFailure(
            f"offline profiler analysis failed with exit status {exc.returncode}: {profile_dir}"
        ) from exc
    print("offline profiler analysis completed", flush=True)


def make_prompt(word_count: int, offset: int = 0) -> str:
    words = (
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mango",
        "november",
        "oscar",
        "papa",
    )
    return " ".join(words[(index + offset) % len(words)] for index in range(word_count))


def _wait_at_barrier(barrier: Barrier, timeout: float, phase: str) -> None:
    try:
        barrier.wait(timeout=timeout)
    except BrokenBarrierError as exc:
        raise SmokeFailure(f"concurrent requests did not reach the {phase} barrier") from exc


def run_streaming_decode(
    args: argparse.Namespace,
    *,
    prompt_offset: int = 0,
    start_barrier: Barrier | None = None,
    profile_start_barrier: Barrier | None = None,
    profile_stop_barrier: Barrier | None = None,
    control_profile: bool = True,
) -> int:
    payload = {
        "model": args.model,
        "prompt": make_prompt(args.prompt_words, prompt_offset),
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        _url(args.base_url, "/v1/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    content_chunks = 0
    profile_start_attempted = False
    profile_started = False
    profile_stop_attempted = False
    profile_stopped = False
    profile_enabled = not args.skip_profile and (control_profile or profile_start_barrier is not None)
    try:
        if start_barrier is not None:
            _wait_at_barrier(start_barrier, args.request_timeout, "start")
        with urllib.request.urlopen(
            request,
            timeout=args.request_timeout,
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise SmokeFailure(f"invalid completion stream event: {data[:200]!r}") from exc
                choices = event.get("choices", [])
                if not any(choice.get("text") for choice in choices):
                    continue
                content_chunks += 1

                if profile_enabled and not profile_started and content_chunks >= args.profile_after_chunks:
                    # A lost HTTP response can hide a successful server-side
                    # start. Remember the attempt first so finally always sends
                    # a best-effort stop in that case.
                    profile_start_attempted = profile_start_barrier is None or control_profile
                    if profile_start_barrier is None:
                        profile_control(
                            args.base_url,
                            "start",
                            args.profile_control_timeout,
                        )
                    else:
                        _wait_at_barrier(
                            profile_start_barrier,
                            args.request_timeout,
                            "profile-start",
                        )
                    profile_started = True
                    if control_profile:
                        print(
                            f"profiler started after {content_chunks} streamed content chunks",
                            flush=True,
                        )

                if (
                    profile_started
                    and not profile_stopped
                    and content_chunks >= args.profile_after_chunks + args.profile_chunks
                ):
                    # torch_npu profiler stop is not safely idempotent. Mark
                    # the attempt before sending it so a lost response does
                    # not cause finally to issue a second stop request.
                    profile_stop_attempted = profile_stop_barrier is None or control_profile
                    if profile_stop_barrier is None:
                        profile_control(
                            args.base_url,
                            "stop",
                            args.profile_control_timeout,
                        )
                    else:
                        _wait_at_barrier(
                            profile_stop_barrier,
                            args.request_timeout,
                            "profile-stop",
                        )
                    profile_stopped = True
                    if control_profile:
                        print(
                            f"profiler stopped after {content_chunks} streamed content chunks",
                            flush=True,
                        )
    finally:
        if profile_start_attempted and not profile_stopped and not profile_stop_attempted:
            try:
                profile_control(
                    args.base_url,
                    "stop",
                    args.profile_control_timeout,
                )
            except Exception as exc:  # best-effort cleanup after a primary failure
                print(f"warning: failed to stop profiler: {exc}", file=sys.stderr)

    minimum_chunks = args.profile_after_chunks + args.profile_chunks if profile_enabled else 2
    if content_chunks < minimum_chunks:
        raise SmokeFailure(f"completion produced only {content_chunks} content chunks; need at least {minimum_chunks}")
    if profile_enabled and not profile_stopped:
        raise SmokeFailure("profile interval did not complete")
    return content_chunks


def run_streaming_decodes(args: argparse.Namespace) -> list[int]:
    if args.concurrency == 1:
        return [run_streaming_decode(args)]
    barrier = Barrier(args.concurrency)
    profile_start_barrier = profile_stop_barrier = None
    if not args.skip_profile:
        profile_start_barrier = Barrier(
            args.concurrency,
            action=lambda: profile_control(
                args.base_url,
                "start",
                args.profile_control_timeout,
            ),
        )
        profile_stop_barrier = Barrier(
            args.concurrency,
            action=lambda: profile_control(
                args.base_url,
                "stop",
                args.profile_control_timeout,
            ),
        )
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_streaming_decode,
                args,
                prompt_offset=index,
                start_barrier=barrier,
                profile_start_barrier=profile_start_barrier,
                profile_stop_barrier=profile_stop_barrier,
                control_profile=index == 0,
            )
            for index in range(args.concurrency)
        ]
        return [future.result() for future in futures]


def check_server_log(path: Path, required_keys: int | None = None) -> int:
    if not path.is_file():
        raise SmokeFailure(f"server log does not exist: {path}")

    capture_complete_seen = False
    expected_layers: int | None = None
    expected_keys: int | None = None
    failures: list[str] = []
    completeness_pattern = re.compile(
        re.escape(_STARTUP_CROSS_LAYER_COMPLETE) + r" for (\d+) local SFA layers and (\d+) keys"
    )

    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            match = completeness_pattern.search(line)
            if match is not None:
                capture_complete_seen = True
                layer_count = int(match.group(1))
                key_count = int(match.group(2))
                if expected_layers is None:
                    expected_layers = layer_count
                elif expected_layers != layer_count:
                    failures.append(
                        f"startup logs disagree on local SFA layer count: {expected_layers} versus {layer_count}"
                    )
                if expected_keys is None:
                    expected_keys = key_count
                elif expected_keys != key_count:
                    failures.append(
                        f"startup logs disagree on staged graph key count: {expected_keys} versus {key_count}"
                    )
            for marker in _FAILURE_MARKERS:
                if marker in line:
                    failures.append(f"line {line_number}: {line.strip()}")

    missing = []
    if not capture_complete_seen:
        missing.append("cross-layer retrieve-split startup capture")
    if expected_layers is None or expected_layers <= 0:
        missing.append("positive local staged-SFA layer count")
    if expected_keys is None or expected_keys <= 0:
        missing.append("positive staged graph key count")
    elif required_keys is not None and expected_keys != required_keys:
        failures.append(f"captured {expected_keys} staged graph keys; expected {required_keys}")
    if missing:
        failures.append("missing log gates: " + ", ".join(missing))
    if failures:
        raise SmokeFailure("server-log validation failed:\n  " + "\n  ".join(failures))

    print(
        "server-log gates passed: cross-layer retrieve-split capture "
        f"({expected_layers} local layers; {expected_keys} keys)"
    )
    assert expected_layers is not None
    return expected_layers


def trace_snapshot(root: Path) -> dict[Path, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        path.resolve(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("trace_view.json")
        if path.is_file()
    }


def wait_for_new_traces(
    root: Path,
    before: dict[Path, tuple[int, int]],
    *,
    expected_ranks: int,
    timeout: float,
) -> list[Path]:
    deadline = time.monotonic() + timeout
    last_states: dict[Path, tuple[int, int]] = {}
    stable_polls = 0
    newest: list[Path] = []
    staged_cache: dict[tuple[Path, tuple[int, int]], bool] = {}
    staged_count = 0
    while time.monotonic() < deadline:
        current = trace_snapshot(root)
        newest = sorted(path for path, state in current.items() if path not in before or before[path] != state)
        states = {path: current[path] for path in newest}
        if len(newest) >= expected_ranks and states == last_states:
            stable_polls += 1
            if stable_polls >= 2:
                staged_count = 0
                for path in newest:
                    cache_key = (path, current[path])
                    if cache_key not in staged_cache:
                        marker_counts = scan_binary(path, _TRACE_MARKERS)
                        staged_cache[cache_key] = all(marker_counts[marker] > 0 for marker in _TRACE_MARKERS)
                    staged_count += int(staged_cache[cache_key])
                if staged_count >= expected_ranks:
                    return newest
                # Files may still be arriving even though the current subset
                # was briefly stable. Avoid rescanning unchanged files.
                stable_polls = 0
        else:
            stable_polls = 0
        last_states = states
        time.sleep(2.0)
    raise SmokeFailure(
        f"found {len(newest)} new trace_view.json files under {root}, but only "
        f"{staged_count} contained all staged ranges; expected at least "
        f"{expected_ranks} staged rank traces within {timeout:g}s"
    )


def scan_binary(path: Path, needles: Iterable[str]) -> dict[str, int]:
    encoded = {needle: needle.encode("utf-8") for needle in needles}
    overlap = max(len(value) for value in encoded.values()) - 1
    counts = {needle: 0 for needle in encoded}
    carry = b""
    with path.open("rb") as trace_file:
        while chunk := trace_file.read(8 * 1024 * 1024):
            data = carry + chunk
            for needle, value in encoded.items():
                counts[needle] += data.count(value)
            carry = data[-overlap:] if overlap else b""
    return counts


def check_traces(paths: list[Path], expected_ranks: int) -> None:
    failures = []
    needles = _TRACE_MARKERS + _LEGACY_GRAPH_MARKERS + _ACL_REPLAY_APIS
    staged_traces: list[tuple[Path, dict[str, int]]] = []
    ignored_traces: list[tuple[Path, list[str]]] = []
    for path in paths:
        counts = scan_binary(path, needles)
        missing_ranges = [marker for marker in _TRACE_MARKERS if counts[marker] == 0]
        if missing_ranges:
            ignored_traces.append((path, missing_ranges))
            continue
        staged_traces.append((path, counts))

    if len(staged_traces) < expected_ranks:
        raise SmokeFailure(
            f"only {len(staged_traces)} of {len(paths)} new traces contained "
            f"all staged ranges; expected at least {expected_ranks}"
        )

    for path, missing_ranges in ignored_traces:
        print(f"ignoring non-staged trace {path}: missing ranges {missing_ranges}")

    for path, counts in staged_traces:
        replay_count = sum(counts[name] for name in _ACL_REPLAY_APIS)
        if replay_count == 0:
            failures.append(f"{path}: no known ACL model-replay API was recorded")
        legacy_count = sum(counts[name] for name in _LEGACY_GRAPH_MARKERS)
        if legacy_count:
            failures.append(f"{path}: found {legacy_count} legacy per-layer graph ranges")
        print(
            f"trace {path}: "
            + ", ".join(f"{marker.rsplit('::', 1)[-1]}={counts[marker]}" for marker in _TRACE_MARKERS)
            + f", ACL replay APIs={replay_count}"
        )
    if failures:
        raise SmokeFailure("trace inventory failed:\n  " + "\n  ".join(failures))

    print(f"validated {len(staged_traces)} staged worker traces; ignored {len(ignored_traces)} extra non-staged traces")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument(
        "--model",
        default="/workspace/models/GLM-5.1-w4a8",
        help="served model name/path used in the OpenAI request",
    )
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=2)
    parser.add_argument("--expected-keys", type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="simultaneous exact-Q1 requests; the first request controls profiling",
    )
    parser.add_argument(
        "--prompt-words",
        type=int,
        default=4096,
        help="common-word prompt; keep its tokenized length >= model index_topk",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--profile-after-chunks", type=int, default=4)
    parser.add_argument("--profile-chunks", type=int, default=12)
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--profile-control-timeout", type=float, default=300)
    parser.add_argument(
        "--profile-analysis-timeout",
        type=float,
        default=900,
        help="timeout for non-daemon offline torch_npu analysis",
    )
    parser.add_argument("--trace-timeout", type=float, default=600)
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="run only startup and request log gates",
    )
    args = parser.parse_args()

    if args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")
    if args.expected_keys is not None and args.expected_keys <= 0:
        parser.error("--expected-keys must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.prompt_words <= 0:
        parser.error("--prompt-words must be positive")
    if args.profile_after_chunks < 3:
        parser.error("profile after at least three chunks to exclude decode warmup")
    if args.profile_chunks <= 0:
        parser.error("--profile-chunks must be positive")
    if args.profile_analysis_timeout <= 0:
        parser.error("--profile-analysis-timeout must be positive")
    required_tokens = args.profile_after_chunks + args.profile_chunks + 2
    if not args.skip_profile and args.max_tokens < required_tokens:
        parser.error(f"--max-tokens must be at least {required_tokens} for this profile interval")
    return args


def main() -> int:
    args = parse_args()
    before_traces = trace_snapshot(args.profile_dir)
    try:
        wait_until_ready(args.base_url, args.ready_timeout)
        if not args.skip_profile:
            require_worker_only_profiling(args.server_log)
        chunks = run_streaming_decodes(args)
        print(f"streaming decodes completed with content chunks per request: {chunks}")
        expected_layers = check_server_log(args.server_log, args.expected_keys)

        if args.skip_profile:
            print("LOG GATES PASSED. Trace and output/TPOT proof were skipped.")
            return 0

        analyse_profile_data(
            args.profile_dir,
            expected_ranks=args.expected_ranks,
            timeout=args.profile_analysis_timeout,
        )
        traces = wait_for_new_traces(
            args.profile_dir,
            before_traces,
            expected_ranks=args.expected_ranks,
            timeout=args.trace_timeout,
        )
        check_traces(traces, args.expected_ranks)
    except (SmokeFailure, urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"STAGED-SFA SMOKE FAILED: {exc}", file=sys.stderr)
        return 1

    print(
        "\nAUTOMATED GATES PASSED; FINAL MINDSTUDIO INSPECTION IS REQUIRED.\n"
        "On one rank and one steady decode step, verify:\n"
        f"  1. sfa_cross_layer::lmcache_retrieve occurs {expected_layers} times "
        "for latent loads, and sfa_cross_layer::bootstrap occurs once.\n"
        "  2. Every retrieval is outside aclmdlRIExecuteAsync graph ranges.\n"
        "  3. An outer graph spans post(layer N), the intervening model ops, "
        "and pre(layer N+1).\n"
        "  4. The captured producer event orders each LMCache load, and the "
        "next-index wait orders the following cross-layer graph.\n"
        "Then compare deterministic tokens and no-profiler TPOT against the "
        "two-graph baseline before accepting the milestone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
