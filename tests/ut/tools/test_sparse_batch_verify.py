# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
from pathlib import Path

from tools import sparse_batch_verify as verify


def _write_server_log(path: Path) -> None:
    path.write_text(
        "[SFA cross-layer graph] captured retrieve-split outer graphs for 2 local SFA layers and 3 keys\n",
        encoding="utf-8",
    )


def _write_trace(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"traceEvents": [{"name": name} for name in names]}),
        encoding="utf-8",
    )


def test_summarize_counts_new_and_legacy_sparse_kernels(tmp_path: Path):
    server_log = tmp_path / "server.log"
    _write_server_log(server_log)
    _write_trace(
        tmp_path / "profile" / "rank0" / "trace_view.json",
        [
            verify._NEW_HOST_OP,
            verify._NEW_HOST_OP,
            "single_layer_paged_kv_copy_v2_mla_sparse_multi_request_bfloat16_t_int64_t",
            "single_layer_paged_kv_copy_v2_mla_sparse_multi_request_bfloat16_t_int64_t",
            verify._RETRIEVE_SCOPE,
        ],
    )

    summary = verify.summarize(tmp_path / "profile", server_log, batch_size=4, log_start_offset=0)

    assert summary["result"] == "PASS"
    assert summary["local_sfa_layers"] == 2
    assert summary["aggregate"]["new_host_op_count"] == 2
    assert summary["aggregate"]["new_device_kernel_count"] == 2
    assert summary["aggregate"]["legacy_device_kernel_count"] == 0
    assert summary["rank_summaries"][0]["new_device_per_local_layer"] == 1


def test_summarize_reports_legacy_kernels_and_new_log_errors(tmp_path: Path):
    server_log = tmp_path / "server.log"
    _write_server_log(server_log)
    start_offset = server_log.stat().st_size
    with server_log.open("a", encoding="utf-8") as log_file:
        log_file.write("507057 DDR address out of range\n")
    _write_trace(
        tmp_path / "profile" / "rank0" / "trace_view.json",
        ["single_layer_paged_kv_copy_v2_mla_sparse_multi_chunk_bfloat16_t_int64_t"],
    )

    summary = verify.summarize(
        tmp_path / "profile",
        server_log,
        batch_size=4,
        log_start_offset=start_offset,
    )

    assert summary["result"] == "FAIL"
    assert summary["aggregate"]["legacy_device_kernel_count"] == 1
    assert summary["server_errors"]["507057"] == 1
    assert any("no multi-request" in failure for failure in summary["failures"])
