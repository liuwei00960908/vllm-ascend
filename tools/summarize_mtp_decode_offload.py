#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize correlated MTP/decode-window diagnostic events from mixed logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MARKER = "[MTP_DW] "
REQUIRED_STAGES = {
    "config",
    "step",
    "meta",
    "finalize",
    "store",
    "commit",
    "remap",
    "retrieve",
    "release",
}


def parse_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Parse schema-1 diagnostic records, ignoring unrelated or malformed lines."""
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        marker_at = line.find(MARKER)
        if marker_at < 0:
            continue
        try:
            event = json.loads(line[marker_at + len(MARKER) :])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict) or event.get("schema") != 1:
            continue
        if not isinstance(event.get("stage"), str):
            continue
        event = dict(event)
        event["_line"] = line_number
        events.append(event)
    return events


def select_request(
    events: list[dict[str, Any]], request_prefix: str | None = None
) -> str | None:
    """Select the latest request matching the optional external-ID prefix."""
    latest_lines: dict[str, int] = {}
    for event in events:
        req = event.get("req")
        if not isinstance(req, str) or not req:
            continue
        if request_prefix is not None and not req.startswith(request_prefix):
            continue
        latest_lines[req] = max(latest_lines.get(req, 0), event["_line"])
    if not latest_lines:
        return None
    return max(latest_lines, key=latest_lines.__getitem__)


def _failure(name: str, event: dict[str, Any], detail: str) -> dict[str, Any]:
    return {
        "invariant": name,
        "line": event.get("_line"),
        "detail": detail,
        "event": event,
    }


def evaluate(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Evaluate locally reconstructable cross-repository invariants."""
    failures: list[dict[str, Any]] = []
    stages = {event["stage"] for event in events}
    missing = sorted(REQUIRED_STAGES - stages)

    config_events = {
        event.get("event")
        for event in events
        if event["stage"] == "config"
    }
    for event_name in ("target_forward", "decode_window"):
        if event_name not in config_events:
            missing.append(f"config:{event_name}")

    for event in events:
        if event["stage"] == "fail":
            failures.append(
                _failure(
                    str(event.get("invariant", "reported_failure")),
                    event,
                    "component reported a locally provable violation",
                )
            )

    for event in (event for event in events if event["stage"] == "step"):
        generated = event.get("generated_count")
        accepted = event.get("accepted_count")
        draft = event.get("draft_count")
        rejected = event.get("rejected_count")
        if not all(
            isinstance(value, int)
            for value in (generated, accepted, draft, rejected)
        ):
            failures.append(
                _failure("mtp_count_fields", event, "missing integer fields")
            )
            continue
        if generated != accepted + 1:
            failures.append(
                _failure(
                    "mtp_generated_count",
                    event,
                    f"{generated} != {accepted} + 1",
                )
            )
        if rejected != draft - accepted:
            failures.append(
                _failure(
                    "mtp_rejected_count",
                    event,
                    f"{rejected} != {draft} - {accepted}",
                )
            )

    for event in (event for event in events if event["stage"] == "meta"):
        start, end = event.get("window_start"), event.get("window_end")
        frontier, size = event.get("frontier"), event.get("window_size")
        if not all(isinstance(value, int) for value in (start, end, frontier, size)):
            failures.append(
                _failure(
                    "synthetic_window_fields", event, "missing integer fields"
                )
            )
        elif end > frontier or end - start != size:
            failures.append(
                _failure(
                    "synthetic_window_frontier",
                    event,
                    f"window=[{start},{end}) frontier={frontier} size={size}",
                )
            )

    finalize_by_frontier: dict[Any, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        if event["stage"] == "finalize" and event.get("event") in {
            "bookkeeping",
            "draft_proposal",
            "connector_finalize",
        }:
            finalize_by_frontier[event.get("frontier")][event["event"]] = event
    required_finalize = {"bookkeeping", "draft_proposal", "connector_finalize"}
    for frontier, finalize in finalize_by_frontier.items():
        if not required_finalize <= finalize.keys():
            missing_finalize = sorted(required_finalize - finalize.keys())
            missing.extend(
                f"finalize:{frontier}:{name}" for name in missing_finalize
            )
            continue
        names = ("bookkeeping", "draft_proposal", "connector_finalize")
        order = [finalize[name].get("order") for name in names]
        line_order = [finalize[name].get("_line") for name in names]
        if (
            order != sorted(order)
            or len(set(order)) != len(order)
            or line_order != sorted(line_order)
        ):
            failures.append(
                _failure(
                    "deferred_finalize_order",
                    finalize["connector_finalize"],
                    f"order={order} lines={line_order}",
                )
            )

    stores: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        if event["stage"] != "store" or event.get("event") != "complete":
            continue
        start, end, group = (
            event.get("window_start"),
            event.get("window_end"),
            event.get("kv_group"),
        )
        if isinstance(start, int) and isinstance(end, int) and isinstance(group, int):
            stores[(start, end)][group] = event
    if stores:
        for window, groups in stores.items():
            if set(groups) != {0, 1}:
                missing.append(f"store_groups:{window[0]}-{window[1]}")
                continue
            actual = {groups[group].get("actual_tokens") for group in (0, 1)}
            expected = window[1] - window[0]
            if actual != {expected}:
                failures.append(
                    _failure(
                        "two_group_store_tokens",
                        groups[0],
                        f"actual={actual} expected={expected}",
                    )
                )
    elif "store" in stages:
        missing.append("store:complete")

    store_group_events = {
        event.get("kv_group")
        for event in events
        if event["stage"] == "store" and event.get("event") == "group_complete"
    }
    if store_group_events != {0, 1}:
        missing.append("store:group_complete")

    previous_commit = -1
    group_complete_lines: dict[tuple[int, int], set[int]] = defaultdict(set)
    for event in events:
        if event["stage"] == "store" and event.get("event") == "group_complete":
            window = (event.get("window_start"), event.get("window_end"))
            group_complete_lines[window].add(event.get("kv_group"))
        if event["stage"] == "commit" and event.get("event") == "frontier_update":
            before, after = event.get("committed_before"), event.get("committed_after")
            if (
                not isinstance(before, int)
                or not isinstance(after, int)
                or after < before
            ):
                failures.append(
                    _failure(
                        "committed_frontier_monotonic",
                        event,
                        f"{before}->{after}",
                    )
                )
            if isinstance(after, int) and after < previous_commit:
                failures.append(
                    _failure(
                        "committed_frontier_global",
                        event,
                        f"{previous_commit}->{after}",
                    )
                )
            if isinstance(after, int):
                previous_commit = after
        if event["stage"] == "commit" and event.get("event") == "publish_completed":
            window = (event.get("window_start"), event.get("window_end"))
            if group_complete_lines.get(window) != {0, 1}:
                failures.append(
                    _failure(
                        "commit_after_required_groups",
                        event,
                        str(group_complete_lines.get(window)),
                    )
                )

    commit_events = {
        event.get("event")
        for event in events
        if event["stage"] == "commit"
    }
    for event_name in ("publish_completed", "frontier_update"):
        if event_name not in commit_events:
            missing.append(f"commit:{event_name}")

    scratch_by_frontier: dict[int, list[int]] = defaultdict(list)
    for event in (event for event in events if event["stage"] == "remap"):
        current, committed, boundary = (
            event.get("window_start"),
            event.get("committed_end"),
            event.get("remap_boundary"),
        )
        if (
            isinstance(current, int)
            and isinstance(committed, int)
            and boundary != min(current, committed)
        ):
            failures.append(
                _failure(
                    "remap_boundary",
                    event,
                    f"{boundary} != min({current},{committed})",
                )
            )
        selected_max = event.get("selected_absolute_max")
        if (
            isinstance(selected_max, int)
            and isinstance(boundary, int)
            and selected_max >= boundary
        ):
            failures.append(
                _failure(
                    "absolute_selected_bound",
                    event,
                    f"{selected_max} >= {boundary}",
                )
            )
        if isinstance(event.get("frontier"), int) and isinstance(
            event.get("scratch_base"), int
        ):
            scratch_by_frontier[event["frontier"]].append(event["scratch_base"])
    for frontier, bases in scratch_by_frontier.items():
        if len(bases) != len(set(bases)):
            event = next(
                event
                for event in events
                if event.get("frontier") == frontier
                and event["stage"] == "remap"
            )
            failures.append(_failure("distinct_mtp_scratch_bases", event, str(bases)))

    for event in (event for event in events if event["stage"] == "retrieve"):
        maximum, available = event.get("selected_max"), event.get("actual_cpu_tokens")
        if event.get("selected_oob") or (
            isinstance(maximum, int) and isinstance(available, int) and maximum >= available
        ):
            failures.append(
                _failure(
                    "retrieve_selected_bound",
                    event,
                    f"max={maximum} available={available}",
                )
            )
        if event.get("selected_count") != event.get("slot_count"):
            failures.append(
                _failure(
                    "retrieve_slot_count",
                    event,
                    "selected/slot counts differ",
                )
            )
        kernel_total = event.get("kernel_total_tokens")
        if (
            isinstance(kernel_total, int)
            and isinstance(available, int)
            and kernel_total != available
        ):
            failures.append(
                _failure(
                    "retrieve_kernel_coverage",
                    event,
                    f"kernel={kernel_total} actual={available}",
                )
            )

    for event in (event for event in events if event["stage"] == "release"):
        start, end = event.get("release_start_block"), event.get("release_end_block")
        scratch, saved_end, block_size = (
            event.get("scratch_blocks"), event.get("saved_end"), event.get("block_size")
        )
        release_fields = (start, end, scratch, saved_end, block_size)
        if all(isinstance(value, int) for value in release_fields):
            if start < scratch or end * block_size > saved_end:
                failures.append(
                    _failure(
                        "release_range",
                        event,
                        f"blocks=[{start},{end}) saved_end={saved_end}",
                    )
                )

    release_events = {
        event.get("event")
        for event in events
        if event["stage"] == "release"
    }
    if "completed_save_consumed" not in release_events:
        missing.append("release:completed_save_consumed")

    unique_missing = list(dict.fromkeys(missing))
    failures.sort(key=lambda failure: failure.get("line") or 0)
    return failures, unique_missing


def summarize_events(
    events: list[dict[str, Any]], request_prefix: str | None = None
) -> dict[str, Any]:
    """Select one request and return its status, failures, and missing stages."""
    request_id = select_request(events, request_prefix)
    if request_id is None:
        return {
            "status": "INCOMPLETE",
            "req": None,
            "events": [],
            "failures": [],
            "missing": ["request"],
        }
    selected = [event for event in events if event.get("req") == request_id]
    failures, missing = evaluate(selected)
    status = "FAIL" if failures else "INCOMPLETE" if missing else "PASS"
    return {
        "status": status,
        "req": request_id,
        "events": selected,
        "failures": failures,
        "missing": missing,
    }


def _format(summary: dict[str, Any], failures_only: bool, stage: str | None) -> str:
    lines = [
        f"{summary['status']} req={summary['req'] or '-'} events={len(summary['events'])}"
    ]
    if summary["missing"]:
        lines.append("missing=" + ",".join(summary["missing"]))
    if summary["failures"]:
        first = summary["failures"][0]
        lines.append(
            f"first_failure={first['invariant']} line={first['line']} {first['detail']}"
        )
    display = summary["events"]
    if stage is not None:
        display = [event for event in display if event["stage"] == stage]
    if failures_only:
        failure_lines = {failure["line"] for failure in summary["failures"]}
        display = [
            event
            for event in display
            if event["_line"] in failure_lines or event["stage"] == "fail"
        ]
    for event in display[:6]:
        bounded = {key: value for key, value in event.items() if key != "_line"}
        lines.append(
            f"line={event['_line']} "
            + json.dumps(bounded, separators=(",", ":"))
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--latest", action="store_true", help="select latest request (default)"
    )
    selection.add_argument("--request-prefix")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--stage", choices=sorted(REQUIRED_STAGES | {"fail"}))
    args = parser.parse_args()

    with args.log.open("r", encoding="utf-8", errors="replace") as log_file:
        events = parse_lines(log_file)
    summary = summarize_events(events, args.request_prefix)
    print(_format(summary, args.failures_only, args.stage))
    if summary["status"] == "FAIL":
        return 1
    if summary["status"] == "INCOMPLETE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
