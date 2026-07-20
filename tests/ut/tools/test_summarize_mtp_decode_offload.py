# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from tools.summarize_mtp_decode_offload import (
    MARKER,
    MAX_EVENT_CHARS,
    _format,
    _format_report,
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
            "freed_blocks": 6,
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


def test_report_identifies_noncommitted_mtp_scratch_target() -> None:
    lines = _lines()
    for tp_rank in range(8):
        lines.append(
            MARKER
            + json.dumps(
                {
                    "schema": 1,
                    "req": "chatcmpl-1-internal",
                    "stage": "deep",
                    "event": "scratch_target_safety",
                    "frontier": 261,
                    "committed_end": 256,
                    "row": 0,
                    "tp_rank": tp_rank,
                    "target_logical_start": 0,
                    "target_logical_end": 256,
                    "boundary": 256,
                    "target_within_committed": True,
                    "target_beyond_current_sequence": False,
                    "target_unmapped_count": 0,
                    "actual_target_live_intersection_count": 0,
                }
            )
        )
    lines.append(
        MARKER
        + json.dumps(
            {
                "schema": 1,
                "req": "chatcmpl-1-internal",
                "stage": "deep",
                "event": "scratch_target_safety",
                "frontier": 262,
                "committed_end": 256,
                "row": 1,
                "target_logical_start": 2048,
                "target_logical_end": 2304,
                "boundary": 256,
                "target_within_committed": False,
                "target_beyond_current_sequence": False,
                "target_unmapped_count": 0,
                "actual_target_live_intersection_count": 256,
            }
        )
    )

    report = _format_report(summarize_events(parse_lines(lines)))

    assert "WINDOW [0,256) store_groups=0,1 committed=yes" in report
    assert report.count("SCRATCH frontier=261 row=0") == 1
    assert "SCRATCH frontier=262 row=1 dest=[2048,2304) boundary=256" in report
    assert "FINDING row1_target_live_alias" in report


def test_report_displays_content_probe_status() -> None:
    lines = _lines()
    lines.extend(
        [
            MARKER
            + json.dumps(
                {
                    "schema": 1,
                    "req": "chatcmpl-1-internal",
                    "stage": "deep",
                    "event": "content_store",
                    "kv_group": 0,
                    "window_end": 256,
                    "chunk_fingerprints": [11, 12],
                }
            ),
            MARKER
            + json.dumps(
                {
                    "schema": 1,
                    "req": "chatcmpl-1-internal",
                    "stage": "deep",
                    "event": "content_transfer",
                    "kv_group": 0,
                    "frontier": 256,
                    "source_chunk_fingerprints": [11, 12],
                    "content_probe": {"supported": True, "all_match": True},
                }
            ),
        ]
    )

    report = _format_report(summarize_events(parse_lines(lines)))

    assert (
        "CONTENT frontier=256 group=0 store_cpu=yes retrieve_cpu=yes "
        "store_retrieve_match=True scatter_supported=True scatter_match=True"
        in report
    )


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


def test_parse_lines_accepts_plain_events_and_lmcache_logger_suffixes() -> None:
    suffixes = {
        "meta": " (vllm_v1_adapter.py:78:lmcache.integration.vllm.vllm_v1_adapter)",
        "store": " (cache_engine.py:47:lmcache_ascend.v1.cache_engine)",
        "commit": " (vllm_v1_adapter.py:78:lmcache.integration.vllm.vllm_v1_adapter)",
        "retrieve": (
            " (npu_connectors.py:57:"
            "lmcache_ascend.v1.npu_connector.npu_connectors)"
        ),
    }
    lines = [MARKER + json.dumps({"schema": 1, "stage": "step", "req": "req"})]
    lines.extend(
        "LMCache INFO: "
        + MARKER
        + json.dumps({"schema": 1, "stage": stage, "req": "req"})
        + suffix
        for stage, suffix in suffixes.items()
    )

    events = parse_lines(lines)

    assert [event["stage"] for event in events] == ["step", *suffixes]
    assert [event["_line"] for event in events] == [1, 2, 3, 4, 5]


def test_parse_lines_ignores_malformed_or_prefixed_payloads() -> None:
    valid = json.dumps({"schema": 1, "stage": "meta", "req": "req"})
    events = parse_lines(
        [
            MARKER + "not-json " + valid,
            MARKER + '{"schema":1,"stage":"meta"',
            MARKER + "[" + valid + "]",
            MARKER + json.dumps({"schema": 2, "stage": "meta"}),
            MARKER + " \t" + valid,
            MARKER + valid + " logger suffix",
        ]
    )

    assert [event["_line"] for event in events] == [5, 6]


def test_complete_mixed_source_summary_accepts_logger_suffixes() -> None:
    logger_suffix_by_stage = {
        "meta": " (vllm_v1_adapter.py:78:lmcache.integration.vllm.vllm_v1_adapter)",
        "store": " (cache_engine.py:47:lmcache_ascend.v1.cache_engine)",
        "commit": " (vllm_v1_adapter.py:78:lmcache.integration.vllm.vllm_v1_adapter)",
        "retrieve": (
            " (npu_connectors.py:57:"
            "lmcache_ascend.v1.npu_connector.npu_connectors)"
        ),
    }
    lines = []
    for line in _lines():
        event = json.loads(line[len(MARKER) :])
        suffix = logger_suffix_by_stage.get(event["stage"], "")
        source_prefix = "LMCache INFO: " if suffix else "Worker INFO: "
        lines.append(source_prefix + line + suffix)

    summary = summarize_events(parse_lines(lines))

    assert summary["status"] == "PASS"
    assert not ({"meta", "store", "commit", "retrieve"} & set(summary["missing"]))


def test_latest_does_not_hide_incomplete_request() -> None:
    events = parse_lines([*_lines("complete"), *_lines("newest")[:-1]])
    summary = summarize_events(events)
    assert summary["status"] == "INCOMPLETE"
    assert summary["req"] == "newest"


def test_latest_uses_request_start_despite_delayed_old_event() -> None:
    old_lines = _lines("old")
    events = parse_lines(
        [*old_lines, *_lines("newest")[:-1], old_lines[-1]]
    )
    summary = summarize_events(events)
    assert summary["status"] == "INCOMPLETE"
    assert summary["req"] == "newest"


def test_finish_and_window_decision_are_not_core_lifecycle_records() -> None:
    lines = _lines()
    auxiliary_events = [
        {
            "schema": 1,
            "req": "chatcmpl-1-internal",
            "stage": "step",
            "event": "request_finish",
            "frontier": 300,
            "output_tokens": 44,
        },
        {
            "schema": 1,
            "req": "chatcmpl-1-internal",
            "stage": "meta",
            "event": "window_decision",
            "frontier": 300,
            "decision": "request_finish",
            "next_start": 256,
            "next_end": 512,
        },
    ]
    lines.extend(MARKER + json.dumps(event) for event in auxiliary_events)
    assert summarize_events(parse_lines(lines))["status"] == "PASS"

    auxiliary_only = summarize_events(
        parse_lines(
            [MARKER + json.dumps(event) for event in auxiliary_events]
        )
    )
    assert auxiliary_only["status"] == "INCOMPLETE"
    assert auxiliary_only["failures"] == []
    assert "step:decode_sample" in auxiliary_only["missing"]
    assert "meta:synthetic_window" in auxiliary_only["missing"]


def test_tp_duplicate_remap_and_finalize_records_are_idempotent() -> None:
    lines = _lines()
    finalize = lines[4:7]
    remap = lines[13]
    del lines[4:7]
    lines[4:4] = [
        *([finalize[2]] * 8),
        *([finalize[0]] * 8),
        *([finalize[1]] * 8),
    ]
    lines.extend([remap] * 7)
    assert summarize_events(parse_lines(lines))["status"] == "PASS"


def test_finalize_explicit_order_is_checked() -> None:
    lines = _lines()
    event = json.loads(lines[5][len(MARKER) :])
    event["order"] = 1
    lines[5] = MARKER + json.dumps(event)
    summary = summarize_events(parse_lines(lines))
    assert summary["status"] == "FAIL"
    assert summary["failures"][0]["invariant"] == "deferred_finalize_order"


def test_finalize_conflicting_tp_order_is_not_hidden_by_valid_replicas() -> None:
    lines = _lines()
    draft = json.loads(lines[5][len(MARKER) :])
    lines.extend([lines[5]] * 7)
    draft["order"] = 3
    lines.append(MARKER + json.dumps(draft))
    summary = summarize_events(parse_lines(lines))
    assert summary["status"] == "FAIL"
    assert any(
        failure["invariant"] == "deferred_finalize_order"
        for failure in summary["failures"]
    )


def test_malformed_finalize_order_is_reported_without_crashing() -> None:
    lines = _lines()
    event = json.loads(lines[5][len(MARKER) :])
    event["order"] = [2]
    lines[5] = MARKER + json.dumps(event)
    summary = summarize_events(parse_lines(lines))
    assert summary["status"] == "FAIL"
    assert summary["failures"][0]["invariant"] == "deferred_finalize_order"


def test_group_completion_can_follow_publish_in_mixed_log() -> None:
    lines = _lines()
    publish = lines.pop(11)
    lines.insert(8, publish)
    assert summarize_events(parse_lines(lines))["status"] == "PASS"


def test_commit_updates_are_not_ordered_across_mixed_process_logs() -> None:
    lines = _lines()
    updates = [
        {
            "schema": 1,
            "req": "chatcmpl-1-internal",
            "stage": "commit",
            "event": "frontier_update",
            "committed_before": 256,
            "committed_after": 512,
        },
        {
            "schema": 1,
            "req": "chatcmpl-1-internal",
            "stage": "commit",
            "event": "frontier_update",
            "committed_before": 0,
            "committed_after": 256,
        },
    ]
    lines.extend(MARKER + json.dumps(event) for event in updates)
    assert summarize_events(parse_lines(lines))["status"] == "PASS"


def test_each_published_window_requires_both_completed_groups() -> None:
    lines = _lines()
    del lines[10]
    summary = summarize_events(parse_lines(lines))
    assert summary["status"] == "FAIL"
    assert any(
        failure["invariant"] == "commit_after_required_groups"
        for failure in summary["failures"]
    )


def test_scratch_reservation_noop_release_is_valid() -> None:
    lines = _lines()
    event = json.loads(lines[15][len(MARKER) :])
    event.update(
        release_start_block=10,
        release_end_block=8,
        scratch_blocks=10,
        freed_blocks=0,
    )
    lines[15] = MARKER + json.dumps(event)
    assert summarize_events(parse_lines(lines))["status"] == "PASS"


def test_noop_release_still_checks_scratch_saved_and_freed_bounds() -> None:
    for changes in (
        {"scratch_blocks": 11},
        {"saved_end": 255},
        {"freed_blocks": 1},
    ):
        lines = _lines()
        event = json.loads(lines[15][len(MARKER) :])
        event.update(
            release_start_block=10,
            release_end_block=8,
            scratch_blocks=10,
            saved_end=256,
            freed_blocks=0,
        )
        event.update(changes)
        lines[15] = MARKER + json.dumps(event)
        summary = summarize_events(parse_lines(lines))
        assert summary["status"] == "FAIL"
        assert any(
            failure["invariant"] == "release_range"
            for failure in summary["failures"]
        )


def test_runtime_fail_event_survives_without_scratch_reconstruction() -> None:
    lines = _lines()
    lines.append(
        MARKER
        + json.dumps(
            {
                "schema": 1,
                "req": "chatcmpl-1-internal",
                "stage": "fail",
                "invariant": "distinct_mtp_scratch_bases",
            }
        )
    )
    summary = summarize_events(parse_lines(lines))
    assert summary["status"] == "FAIL"
    assert any(
        failure["invariant"] == "distinct_mtp_scratch_bases"
        for failure in summary["failures"]
    )


def test_formatted_event_context_is_bounded() -> None:
    lines = _lines()
    event = json.loads(lines[13][len(MARKER) :])
    event["bounded_sample"] = list(range(1000))
    lines[13] = MARKER + json.dumps(event)
    summary = summarize_events(parse_lines(lines))
    rendered = _format(summary, failures_only=False, stage="remap")
    assert len(rendered.splitlines()[-1]) <= MAX_EVENT_CHARS + 20


def test_deep_events_parse_filter_without_becoming_required() -> None:
    lines = _lines()
    lines.append(
        MARKER
        + json.dumps(
            {
                "schema": 1,
                "req": "chatcmpl-1-internal",
                "stage": "deep",
                "event": "row_mapping",
                "selected_absolute_checksum": 17,
            }
        )
    )

    summary = summarize_events(parse_lines(lines))
    rendered = _format(summary, failures_only=False, stage="deep")
    assert summary["status"] == "PASS"
    assert '"stage":"deep"' in rendered
    assert '"stage":"remap"' not in rendered


def test_deep_alias_failure_is_reported() -> None:
    lines = _lines()
    lines.append(
        MARKER
        + json.dumps(
            {
                "schema": 1,
                "req": "chatcmpl-1-internal",
                "stage": "fail",
                "invariant": "scratch_live_slot_alias",
                "intersection_count": 1,
                "intersection_sample": [128],
            }
        )
    )

    summary = summarize_events(parse_lines(lines))
    assert summary["status"] == "FAIL"
    assert summary["failures"][-1]["invariant"] == "scratch_live_slot_alias"
