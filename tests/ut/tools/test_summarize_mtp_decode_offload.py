# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from tools.summarize_mtp_decode_offload import (
    MARKER,
    parse_lines,
    summarize_events,
)


def _lines(req: str = "chatcmpl-1-internal") -> list[str]:
    events = [
        {"stage": "config", "event": "target_forward"},
        {"stage": "config", "event": "decode_window"},
        {
            "stage": "step",
            "draft_count": 2,
            "accepted_count": 1,
            "rejected_count": 1,
            "generated_count": 2,
        },
        {
            "stage": "meta",
            "frontier": 300,
            "window_start": 0,
            "window_end": 256,
            "window_size": 256,
        },
        {
            "stage": "finalize",
            "event": "bookkeeping",
            "frontier": 255,
            "order": 1,
        },
        {
            "stage": "finalize",
            "event": "draft_proposal",
            "frontier": 255,
            "order": 2,
        },
        {
            "stage": "finalize",
            "event": "connector_finalize",
            "frontier": 255,
            "order": 3,
        },
        {
            "stage": "store",
            "event": "complete",
            "window_start": 0,
            "window_end": 256,
            "kv_group": 0,
            "actual_tokens": 256,
        },
        {
            "stage": "store",
            "event": "complete",
            "window_start": 0,
            "window_end": 256,
            "kv_group": 1,
            "actual_tokens": 256,
        },
        {
            "stage": "store",
            "event": "group_complete",
            "window_start": 0,
            "window_end": 256,
            "kv_group": 0,
        },
        {
            "stage": "store",
            "event": "group_complete",
            "window_start": 0,
            "window_end": 256,
            "kv_group": 1,
        },
        {
            "stage": "commit",
            "event": "publish_completed",
            "window_start": 0,
            "window_end": 256,
        },
        {
            "stage": "commit",
            "event": "frontier_update",
            "committed_before": 0,
            "committed_after": 256,
        },
        {
            "stage": "remap",
            "frontier": 257,
            "window_start": 256,
            "committed_end": 256,
            "remap_boundary": 256,
            "scratch_base": 0,
            "selected_absolute_max": 255,
        },
        {
            "stage": "retrieve",
            "actual_cpu_tokens": 256,
            "selected_count": 2,
            "slot_count": 2,
            "selected_max": 255,
            "selected_oob": False,
            "kernel_total_tokens": 256,
        },
        {
            "stage": "release",
            "release_start_block": 2,
            "release_end_block": 8,
            "scratch_blocks": 2,
            "saved_end": 256,
            "block_size": 32,
        },
        {
            "stage": "release",
            "event": "completed_save_consumed",
            "saved_end": 256,
            "freed_blocks": 6,
        },
    ]
    return [
        MARKER + json.dumps({"schema": 1, "req": req, **event})
        for event in events
    ]


def test_pass_and_request_prefix() -> None:
    events = parse_lines([*_lines("old"), *_lines("chatcmpl-http-worker")])
    summary = summarize_events(events, "chatcmpl-http")
    assert summary["status"] == "PASS"
    assert summary["req"] == "chatcmpl-http-worker"


def test_failure_reports_first_invariant() -> None:
    lines = _lines()
    event = json.loads(lines[2][len(MARKER) :])
    event["generated_count"] = 3
    lines[2] = MARKER + json.dumps(event)
    summary = summarize_events(parse_lines(lines))
    assert summary["status"] == "FAIL"
    assert summary["failures"][0]["invariant"] == "mtp_generated_count"


def test_missing_stage_is_incomplete() -> None:
    summary = summarize_events(
        parse_lines(
            [
                line
                for line in _lines()
                if json.loads(line[len(MARKER) :])["stage"] != "release"
            ]
        )
    )
    assert summary["status"] == "INCOMPLETE"
    assert "release" in summary["missing"]


def test_latest_and_malformed_lines() -> None:
    events = parse_lines(
        ["noise", MARKER + "not-json", *_lines("first"), *_lines("latest")]
    )
    summary = summarize_events(events)
    assert summary["status"] == "PASS"
    assert summary["req"] == "latest"


def test_latest_does_not_hide_incomplete_request() -> None:
    events = parse_lines([*_lines("complete"), *_lines("newest")[:-1]])
    summary = summarize_events(events)
    assert summary["status"] == "INCOMPLETE"
    assert summary["req"] == "newest"


def test_finalize_source_order_is_checked() -> None:
    lines = _lines()
    lines[4], lines[5] = lines[5], lines[4]
    summary = summarize_events(parse_lines(lines))
    assert summary["status"] == "FAIL"
    assert summary["failures"][0]["invariant"] == "deferred_finalize_order"
