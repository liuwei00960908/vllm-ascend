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
MAX_DISPLAY_EVENTS = 6
MAX_EVENT_CHARS = 1000
MAX_DETAIL_CHARS = 500
MAX_MISSING_ITEMS = 20
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
    decoder = json.JSONDecoder()
    for line_number, line in enumerate(lines, 1):
        marker_at = line.find(MARKER)
        if marker_at < 0:
            continue
        payload = line[marker_at + len(MARKER) :].lstrip()
        if not payload.startswith("{"):
            continue
        try:
            event, _ = decoder.raw_decode(payload)
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
    first_lines: dict[str, int] = {}
    for event in events:
        req = event.get("req")
        if not isinstance(req, str) or not req:
            continue
        if request_prefix is not None and not req.startswith(request_prefix):
            continue
        first_lines.setdefault(req, event["_line"])
    if not first_lines:
        return None
    return max(first_lines, key=first_lines.__getitem__)


def _failure(name: str, event: dict[str, Any], detail: str) -> dict[str, Any]:
    if len(detail) > MAX_DETAIL_CHARS:
        detail = detail[: MAX_DETAIL_CHARS - 3] + "..."
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

    step_events = [
        event
        for event in events
        if event["stage"] == "step" and event.get("event") != "request_finish"
    ]
    if "step" in stages and not step_events:
        missing.append("step:decode_sample")
    for event in step_events:
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

    meta_events = [
        event
        for event in events
        if event["stage"] == "meta" and event.get("event") != "window_decision"
    ]
    if "meta" in stages and not meta_events:
        missing.append("meta:synthetic_window")
    for event in meta_events:
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

    finalize_by_frontier: dict[Any, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        if event["stage"] == "finalize" and event.get("event") in {
            "bookkeeping",
            "draft_proposal",
            "connector_finalize",
        }:
            frontier = event.get("frontier")
            if not isinstance(frontier, int):
                failures.append(
                    _failure(
                        "finalize_fields",
                        event,
                        "frontier must be an integer",
                    )
                )
                continue
            finalize_by_frontier[frontier][event["event"]].append(event)
    required_finalize = {"bookkeeping", "draft_proposal", "connector_finalize"}
    expected_finalize_order = {
        "bookkeeping": 1,
        "draft_proposal": 2,
        "connector_finalize": 3,
    }
    if "finalize" in stages and not finalize_by_frontier:
        missing.extend(
            f"finalize:{name}" for name in sorted(required_finalize)
        )
    for frontier, finalize in finalize_by_frontier.items():
        if not required_finalize <= finalize.keys():
            missing_finalize = sorted(required_finalize - finalize.keys())
            missing.extend(
                f"finalize:{frontier}:{name}" for name in missing_finalize
            )
            continue
        for name, expected_order in expected_finalize_order.items():
            observed_orders = [event.get("order") for event in finalize[name]]
            if not all(order == expected_order for order in observed_orders):
                failures.append(
                    _failure(
                        "deferred_finalize_order",
                        finalize[name][0],
                        f"frontier={frontier} event={name} "
                        f"order={sorted(observed_orders, key=str)} "
                        f"expected={expected_order}",
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

    completed_groups: dict[tuple[int, int], set[int]] = defaultdict(set)
    for event in events:
        if event["stage"] == "store" and event.get("event") == "group_complete":
            start, end, group = (
                event.get("window_start"),
                event.get("window_end"),
                event.get("kv_group"),
            )
            if all(isinstance(value, int) for value in (start, end, group)):
                completed_groups[(start, end)].add(group)
    for event in events:
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
        if event["stage"] == "commit" and event.get("event") == "publish_completed":
            start, end = event.get("window_start"), event.get("window_end")
            if not isinstance(start, int) or not isinstance(end, int):
                failures.append(
                    _failure(
                        "commit_fields",
                        event,
                        "published window bounds must be integers",
                    )
                )
                continue
            window = (start, end)
            if completed_groups.get(window) != {0, 1}:
                failures.append(
                    _failure(
                        "commit_after_required_groups",
                        event,
                        str(completed_groups.get(window)),
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
            freed = event.get("freed_blocks")
            invalid_noop = end <= start and isinstance(freed, int) and freed != 0
            if (
                min(start, end, scratch, saved_end) < 0
                or block_size <= 0
                or start < scratch
                or end * block_size > saved_end
                or invalid_noop
            ):
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
        displayed_missing = summary["missing"][:MAX_MISSING_ITEMS]
        missing_text = ",".join(displayed_missing)
        if len(summary["missing"]) > MAX_MISSING_ITEMS:
            missing_text += f",...(+{len(summary['missing']) - MAX_MISSING_ITEMS})"
        lines.append("missing=" + missing_text)
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
    for event in display[:MAX_DISPLAY_EVENTS]:
        bounded = {key: value for key, value in event.items() if key != "_line"}
        rendered = json.dumps(bounded, separators=(",", ":"))
        if len(rendered) > MAX_EVENT_CHARS:
            rendered = rendered[: MAX_EVENT_CHARS - 3] + "..."
        lines.append(f"line={event['_line']} {rendered}")
    return "\n".join(lines)


def _format_report(summary: dict[str, Any]) -> str:
    """Render a compact, pasteable per-request diagnostic report."""
    lines = [
        f"REQ {summary['req'] or '-'} status={summary['status']} "
        f"events={len(summary['events'])}"
    ]
    if summary["missing"]:
        lines.append("MISSING " + ",".join(summary["missing"][:MAX_MISSING_ITEMS]))

    events = summary["events"]
    stores: dict[tuple[int, int], set[int]] = defaultdict(set)
    committed: set[tuple[int, int]] = set()
    retrieve_groups: set[int] = set()
    safety_events: list[dict[str, Any]] = []
    store_content: dict[tuple[int, int], dict[str, Any]] = {}
    transfer_content: dict[tuple[int, int], dict[str, Any]] = {}
    for event in events:
        window = (event.get("window_start"), event.get("window_end"))
        if (
            event.get("stage") == "store"
            and event.get("event") == "complete"
            and all(isinstance(value, int) for value in window)
            and isinstance(event.get("kv_group"), int)
        ):
            stores[window].add(event["kv_group"])
        if (
            event.get("stage") == "commit"
            and event.get("event") == "publish_completed"
            and all(isinstance(value, int) for value in window)
        ):
            committed.add(window)
        if event.get("stage") == "retrieve" and isinstance(
            event.get("kv_group"), int
        ):
            retrieve_groups.add(event["kv_group"])
        if (
            event.get("stage") == "deep"
            and event.get("event") == "scratch_target_safety"
        ):
            safety_events.append(event)
        if (
            event.get("stage") == "deep"
            and event.get("event") == "content_store"
            and isinstance(event.get("kv_group"), int)
            and isinstance(event.get("window_end"), int)
        ):
            store_content.setdefault((event["window_end"], event["kv_group"]), event)
        if (
            event.get("stage") == "deep"
            and event.get("event") == "content_transfer"
            and isinstance(event.get("kv_group"), int)
            and isinstance(event.get("frontier"), int)
        ):
            transfer_content.setdefault((event["frontier"], event["kv_group"]), event)
        if (
            event.get("stage") == "deep"
            and event.get("event") == "content_skip"
            and isinstance(event.get("kv_group"), int)
            and isinstance(event.get("frontier"), int)
        ):
            key = (event["frontier"], event["kv_group"])
            if key not in transfer_content:
                transfer_content[key] = event

    windows = sorted(set(stores) | committed)
    for start, end in windows[:6]:
        groups = ",".join(str(group) for group in sorted(stores[(start, end)])) or "-"
        lines.append(
            f"WINDOW [{start},{end}) store_groups={groups} "
            f"committed={'yes' if (start, end) in committed else 'no'}"
        )
    groups = ",".join(str(group) for group in sorted(retrieve_groups)) or "-"
    lines.append(f"RETRIEVE groups={groups}")
    for frontier, group in sorted(set(store_content) | set(transfer_content)):
        store_event = store_content.get((frontier, group))
        transfer_event = transfer_content.get((frontier, group))
        store_ranges = (
            {
                (chunk.get("start"), chunk.get("end")): chunk.get("fingerprint")
                for chunk in store_event.get("chunk_ranges", [])
                if isinstance(chunk, dict)
            }
            if store_event is not None
            else {}
        )
        retrieve_ranges = (
            {
                (chunk.get("start"), chunk.get("end")): chunk.get("fingerprint")
                for chunk in transfer_event.get("source_chunk_ranges", [])
                if isinstance(chunk, dict)
            }
            if transfer_event is not None
            else {}
        )
        overlapping = sorted(set(store_ranges) & set(retrieve_ranges))
        if overlapping:
            mismatches = [
                f"[{s},{e})"
                for (s, e) in overlapping
                if store_ranges[(s, e)] != retrieve_ranges[(s, e)]
            ]
            source_match = len(mismatches) == 0
            source_detail = (
                f"matched={len(overlapping)}"
                + (f" mismatch={mismatches[:2]}" if mismatches else "")
                + (f" overlap={overlapping[:2]}" if not mismatches else "")
            )
        else:
            source_match = None
            source_detail = "no_overlap"
        probe = transfer_event.get("content_probe", {}) if transfer_event else {}
        skip_reason = (
            transfer_event.get("reason")
            if transfer_event and transfer_event.get("event") == "content_skip"
            else None
        )
        lines.append(
            f"CONTENT frontier={frontier} group={group} "
            f"store_cpu={'yes' if store_event else 'no'} "
            f"retrieve_cpu={'yes' if transfer_event and transfer_event.get('event') == 'content_transfer' else 'no'} "
            f"store_ranges={sorted(store_ranges)[:6]} "
            f"retrieve_ranges={sorted(retrieve_ranges)[:6]} "
            f"store_retrieve_match={source_match} {source_detail} "
            f"scatter_supported={probe.get('supported')} "
            f"scatter_match={probe.get('all_match')}"
            + (f" skip={skip_reason}" if skip_reason else "")
        )

    safety_by_row: dict[tuple[int, int, int], dict[str, Any]] = {}
    for event in safety_events:
        key = (
            int(event.get("boundary", -1) or -1),
            int(event.get("committed_end", -1) or -1),
            int(event.get("row", -1) or -1),
        )
        safety_by_row.setdefault(key, event)
    unique_safety_events = [
        safety_by_row[key] for key in sorted(safety_by_row)
    ]

    findings: list[str] = []
    for event in unique_safety_events:
        row = event.get("row")
        live_aliases = event.get("actual_target_live_intersection_count")
        unmapped = event.get("target_unmapped_count")
        if isinstance(unmapped, int) and unmapped > 0:
            findings.append(f"row{row}_target_unmapped")
        if isinstance(live_aliases, int) and live_aliases > 0:
            findings.append(f"row{row}_target_live_alias")
        oor = event.get("target_blocks_out_of_range")
        if isinstance(oor, int) and oor > 0:
            findings.append(f"row{row}_blocks_out_of_range")
        if event.get("target_blocks_in_capacity") is False:
            findings.append(f"row{row}_target_block_oob")

    for event in unique_safety_events[:8]:
        row = event.get("row")
        start = event.get("target_logical_start")
        end = event.get("target_logical_end")
        boundary = event.get("boundary")
        within = event.get("target_within_committed")
        beyond = event.get("target_beyond_current_sequence")
        live_aliases = event.get("actual_target_live_intersection_count")
        unmapped = event.get("target_unmapped_count")
        num_blocks = event.get("num_blocks")
        blocks_oor = event.get("target_blocks_out_of_range")
        capacity = event.get("physical_block_capacity")
        target_blocks = event.get("target_physical_block_ids")
        target_in_capacity = event.get("target_blocks_in_capacity")
        lines.append(
            f"SCRATCH frontier={event.get('frontier')} row={row} "
            f"dest=[{start},{end}) boundary={boundary} "
            f"within_committed={within} beyond_sequence={beyond} "
            f"unmapped={unmapped} live_aliases={live_aliases} "
            f"num_blocks={num_blocks} blocks_oor={blocks_oor} "
            f"capacity={capacity} target_blocks={target_blocks} "
            f"target_in_capacity={target_in_capacity}"
        )
    if not unique_safety_events:
        findings.append("scratch_target_safety_missing")
    if findings:
        lines.append("FINDING " + ",".join(dict.fromkeys(findings)))
    else:
        lines.append("FINDING no_scratch_target_violation_reported")
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
    parser.add_argument(
        "--report",
        action="store_true",
        help="print a compact per-request lifecycle and scratch safety report",
    )
    parser.add_argument(
        "--stage", choices=sorted(REQUIRED_STAGES | {"deep", "fail"})
    )
    args = parser.parse_args()

    with args.log.open("r", encoding="utf-8", errors="replace") as log_file:
        events = parse_lines(log_file)
    summary = summarize_events(events, args.request_prefix)
    if args.report:
        print(_format_report(summary))
    else:
        print(_format(summary, args.failures_only, args.stage))
    if summary["status"] == "FAIL":
        return 1
    if summary["status"] == "INCOMPLETE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
