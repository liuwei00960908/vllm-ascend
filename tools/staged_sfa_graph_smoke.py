#!/usr/bin/env python3
"""Drive and check the narrow TP8 staged-SFA graph proof of concept.

The server must already be running with TP=8, no speculative decoding,
SHRINK_LATENT=2, PIECEWISE graph mode, and the staged-SFA POC enabled.  This
client sends one long, streaming completion.  It starts the online Ascend
PyTorch profiler only after the first four streamed token chunks so the first
two eager-vs-graph parity steps are normally outside the steady-state trace.

Automated checks require the model-level startup capture and ordered
replay-canary completeness check and two live numerical parity steps to pass.
They also require at least eight new staged worker traces to contain all three
ranges plus an ACL model-replay API. They do not prove timeline nesting or
device execution density; finish the checklist below in MindStudio Insight
before calling the hardware proof successful.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


_TRACE_MARKERS = (
    "sfa_staged_graph_poc::pre",
    "sfa_staged_graph_poc::lmcache_retrieve",
    "sfa_staged_graph_poc::post",
)
_PARITY_MARKERS = (
    "sfa_staged_graph_poc::live_parity_pre",
    "sfa_staged_graph_poc::live_parity_post",
)
_ACL_REPLAY_APIS = (
    "aclmdlRIExecuteAsync",
    "aclmdlExecuteAsync",
    "aclmdlExecuteAsyncV2",
)
_STARTUP_REPLAY_CANARY_COMPLETE = (
    "[SFA staged graph POC] startup capture and ordered replay-canary "
    "completeness check passed"
)
_LIVE_SIGNATURE_VALIDATION = (
    "[SFA staged graph POC] verified pre/post startup capture and enabled "
    "always-on captured-input signature validation for live replay; LMCache "
    "retrieval remains eager."
)
_PARITY_PASS = "[SFA staged graph POC] live eager-vs-graph parity passed"
_FAILURE_MARKERS = (
    "[SFA staged graph POC] live eager-vs-graph parity failed",
    "[SFA staged graph POC] startup capture is incomplete",
    "[SFA staged graph POC] startup ordered replay-canary check failed",
    "[SFA staged graph POC] startup ordered replay-canary check is incomplete",
    "[SFA staged graph POC] the one-token dummy pass is ineligible",
    "positional tensor storage for the pre graph changed",
    "positional tensor storage for the post graph changed",
    "full tensor signature for the",
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
    raise SmokeFailure(
        f"server did not become healthy within {timeout:g}s: {last_error}"
    )


def profile_control(base_url: str, action: str, timeout: float) -> None:
    with _request(
        _url(base_url, f"/{action}_profile"),
        method="POST",
        timeout=timeout,
    ) as response:
        body = response.read().decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise SmokeFailure(
                f"{action}_profile returned HTTP {response.status}: {body}"
            )


def make_prompt(word_count: int) -> str:
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
    return " ".join(words[index % len(words)] for index in range(word_count))


def run_streaming_decode(args: argparse.Namespace) -> int:
    payload = {
        "model": args.model,
        "prompt": make_prompt(args.prompt_words),
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
    profile_stopped = False
    try:
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
                    raise SmokeFailure(
                        f"invalid completion stream event: {data[:200]!r}"
                    ) from exc
                choices = event.get("choices", [])
                if not any(choice.get("text") for choice in choices):
                    continue
                content_chunks += 1

                if (
                    not args.skip_profile
                    and not profile_started
                    and content_chunks >= args.profile_after_chunks
                ):
                    # A lost HTTP response can hide a successful server-side
                    # start. Remember the attempt first so finally always sends
                    # a best-effort stop in that case.
                    profile_start_attempted = True
                    profile_control(
                        args.base_url,
                        "start",
                        args.profile_control_timeout,
                    )
                    profile_started = True
                    print(
                        "profiler started after "
                        f"{content_chunks} streamed content chunks",
                        flush=True,
                    )

                if (
                    profile_started
                    and not profile_stopped
                    and content_chunks
                    >= args.profile_after_chunks + args.profile_chunks
                ):
                    profile_control(
                        args.base_url,
                        "stop",
                        args.profile_control_timeout,
                    )
                    profile_stopped = True
                    print(
                        "profiler stopped after "
                        f"{content_chunks} streamed content chunks",
                        flush=True,
                    )
    finally:
        if profile_start_attempted and not profile_stopped:
            try:
                profile_control(
                    args.base_url,
                    "stop",
                    args.profile_control_timeout,
                )
            except Exception as exc:  # best-effort cleanup after a primary failure
                print(f"warning: failed to stop profiler: {exc}", file=sys.stderr)

    minimum_chunks = (
        args.profile_after_chunks + args.profile_chunks
        if not args.skip_profile
        else 2
    )
    if content_chunks < minimum_chunks:
        raise SmokeFailure(
            f"completion produced only {content_chunks} content chunks; "
            f"need at least {minimum_chunks}"
        )
    if not args.skip_profile and not profile_stopped:
        raise SmokeFailure("profile interval did not complete")
    return content_chunks


def check_server_log(path: Path) -> int:
    if not path.is_file():
        raise SmokeFailure(f"server log does not exist: {path}")

    replay_canary_complete_seen = False
    signature_seen = False
    first_parity_seen = False
    second_parity_seen = False
    parity_passes = 0
    expected_layers: int | None = None
    expected_graphs: int | None = None
    failures: list[str] = []
    completeness_pattern = re.compile(
        re.escape(_STARTUP_REPLAY_CANARY_COMPLETE)
        + r" for (\d+) local SFA layers \((\d+) staged graphs\)\."
    )

    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            signature_seen |= _LIVE_SIGNATURE_VALIDATION in line
            if _PARITY_PASS in line:
                parity_passes += 1
                first_parity_seen |= "(1/2 live lengths" in line
                second_parity_seen |= "(2/2 live lengths" in line
            match = completeness_pattern.search(line)
            if match is not None:
                replay_canary_complete_seen = True
                layer_count = int(match.group(1))
                graph_count = int(match.group(2))
                if expected_layers is None:
                    expected_layers = layer_count
                elif expected_layers != layer_count:
                    failures.append(
                        "startup logs disagree on local SFA layer count: "
                        f"{expected_layers} versus {layer_count}"
                    )
                if expected_graphs is None:
                    expected_graphs = graph_count
                elif expected_graphs != graph_count:
                    failures.append(
                        "startup logs disagree on staged graph count: "
                        f"{expected_graphs} versus {graph_count}"
                    )
            for marker in _FAILURE_MARKERS:
                if marker in line:
                    failures.append(f"line {line_number}: {line.strip()}")

    missing = []
    if not replay_canary_complete_seen:
        missing.append(
            "model-level startup capture and ordered replay-canary "
            "completeness check"
        )
    if not signature_seen:
        missing.append("always-on live captured-input signature validation")
    if not first_parity_seen:
        missing.append("first live eager-vs-graph parity pass")
    if not second_parity_seen:
        missing.append("second distinct-length eager-vs-graph parity pass")
    if expected_layers is None or expected_layers <= 0:
        missing.append("positive local staged-SFA layer count")
    if expected_graphs is None or expected_graphs <= 0:
        missing.append("positive staged graph count")
    elif expected_layers is not None and expected_graphs != 2 * expected_layers:
        failures.append(
            "startup completeness marker reported "
            f"{expected_graphs} staged graphs for {expected_layers} local SFA "
            "layers; expected exactly one pre and one post graph per layer"
        )
    if missing:
        failures.append("missing log gates: " + ", ".join(missing))
    if failures:
        raise SmokeFailure("server-log validation failed:\n  " + "\n  ".join(failures))

    print(
        "server-log gates passed: model-level startup capture and ordered "
        "replay-canary completeness, "
        "always-on live input-signature validation, and two parity lengths "
        f"({expected_layers} local layers; {expected_graphs} staged graphs; "
        f"{parity_passes} rank-visible "
        "parity messages)"
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
        newest = sorted(
            path
            for path, state in current.items()
            if path not in before or before[path] != state
        )
        states = {path: current[path] for path in newest}
        if len(newest) >= expected_ranks and states == last_states:
            stable_polls += 1
            if stable_polls >= 2:
                staged_count = 0
                for path in newest:
                    cache_key = (path, current[path])
                    if cache_key not in staged_cache:
                        marker_counts = scan_binary(path, _TRACE_MARKERS)
                        staged_cache[cache_key] = all(
                            marker_counts[marker] > 0
                            for marker in _TRACE_MARKERS
                        )
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
    needles = _TRACE_MARKERS + _PARITY_MARKERS + _ACL_REPLAY_APIS
    staged_traces: list[tuple[Path, dict[str, int]]] = []
    ignored_traces: list[tuple[Path, list[str]]] = []
    for path in paths:
        counts = scan_binary(path, needles)
        missing_ranges = [
            marker for marker in _TRACE_MARKERS if counts[marker] == 0
        ]
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
        print(
            f"ignoring non-staged trace {path}: missing ranges "
            f"{missing_ranges}"
        )

    for path, counts in staged_traces:
        replay_count = sum(counts[name] for name in _ACL_REPLAY_APIS)
        parity_count = sum(counts[name] for name in _PARITY_MARKERS)
        if replay_count == 0:
            failures.append(f"{path}: no known ACL model-replay API was recorded")
        if parity_count:
            failures.append(
                f"{path}: steady-state profile contains {parity_count} live-parity "
                "ranges; start profiling later"
            )
        print(
            f"trace {path}: "
            + ", ".join(
                f"{marker.rsplit('::', 1)[-1]}={counts[marker]}"
                for marker in _TRACE_MARKERS
            )
            + f", ACL replay APIs={replay_count}"
        )
    if failures:
        raise SmokeFailure("trace inventory failed:\n  " + "\n  ".join(failures))

    print(
        f"validated {len(staged_traces)} staged worker traces; "
        f"ignored {len(ignored_traces)} extra non-staged traces"
    )


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
    parser.add_argument("--expected-ranks", type=int, default=8)
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
    parser.add_argument("--trace-timeout", type=float, default=600)
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="run only startup/live-parity log gates",
    )
    args = parser.parse_args()

    if args.expected_ranks != 8:
        parser.error("this POC smoke is intentionally restricted to TP8")
    if args.prompt_words <= 0:
        parser.error("--prompt-words must be positive")
    if args.profile_after_chunks < 3:
        parser.error("profile after at least three chunks to exclude parity steps")
    if args.profile_chunks <= 0:
        parser.error("--profile-chunks must be positive")
    required_tokens = args.profile_after_chunks + args.profile_chunks + 2
    if not args.skip_profile and args.max_tokens < required_tokens:
        parser.error(
            f"--max-tokens must be at least {required_tokens} for this profile interval"
        )
    return args


def main() -> int:
    args = parse_args()
    before_traces = trace_snapshot(args.profile_dir)
    try:
        wait_until_ready(args.base_url, args.ready_timeout)
        chunks = run_streaming_decode(args)
        print(f"streaming decode completed with {chunks} content chunks")
        expected_layers = check_server_log(args.server_log)

        if args.skip_profile:
            print(
                "LOG GATES PASSED. Trace proof was skipped; capture success is "
                "not yet established."
            )
            return 0

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
        "On one rank and one steady decode step, verify for every SFA layer:\n"
        "  1. sfa_staged_graph_poc::pre contains one aclmdlRIExecuteAsync "
        "(or version-equivalent ACL model replay), not individual eager op launches.\n"
        "  2. sfa_staged_graph_poc::lmcache_retrieve is outside both graph ranges.\n"
        "  3. sfa_staged_graph_poc::post contains a second ACL model replay.\n"
        "  4. The two replay calls each drive their corresponding NPU compute "
        "island, with LMCache stream/event work between them.\n"
        f"  5. pre and post each occur about {expected_layers} times per decoded "
        "token on that rank; no live_parity ranges occur in the measured interval.\n"
        "Only after those timeline relationships are visible is the TP8 hardware "
        "capture/replay proof complete."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
