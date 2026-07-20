# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import staged_sfa_graph_smoke as smoke


class _StreamingResponse:
    def __init__(self, chunks: int):
        event = b'data: {"choices": [{"text": "token"}]}\n'
        self._lines = [event] * chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)


def _decode_args() -> SimpleNamespace:
    return SimpleNamespace(
        base_url="http://127.0.0.1:9000",
        model="model",
        prompt_words=1,
        max_tokens=4,
        request_timeout=5,
        skip_profile=False,
        profile_after_chunks=1,
        profile_chunks=1,
        profile_control_timeout=5,
    )


def test_stop_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        smoke.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _StreamingResponse(chunks=2),
    )
    actions = []

    def profile_control(base_url, action, timeout):
        actions.append(action)
        if action == "stop":
            raise urllib.error.URLError("response lost")

    monkeypatch.setattr(smoke, "profile_control", profile_control)

    with pytest.raises(urllib.error.URLError, match="response lost"):
        smoke.run_streaming_decode(_decode_args())

    assert actions == ["start", "stop"]


def test_start_failure_still_gets_one_cleanup_stop(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        smoke.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _StreamingResponse(chunks=1),
    )
    actions = []

    def profile_control(base_url, action, timeout):
        actions.append(action)
        if action == "start":
            raise urllib.error.URLError("response lost")

    monkeypatch.setattr(smoke, "profile_control", profile_control)

    with pytest.raises(urllib.error.URLError, match="response lost"):
        smoke.run_streaming_decode(_decode_args())

    assert actions == ["start", "stop"]


def test_offline_analysis_runs_in_bounded_subprocess(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(smoke.subprocess, "run", run)
    profile_dir = Path("/profiles/tp8")

    smoke.analyse_profile_data(profile_dir, expected_ranks=8, timeout=17)

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == sys.executable
    assert command[1] == "-c"
    assert "torch_npu.profiler.profiler import analyse" in command[2]
    assert command[3:] == [str(profile_dir), "8"]
    assert "max_process_number=int(sys.argv[2])" in command[2]
    assert kwargs == {"check": True, "timeout": 17}


def test_offline_analysis_timeout_is_a_smoke_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(smoke.subprocess, "run", run)

    with pytest.raises(smoke.SmokeFailure, match="did not finish within 3s"):
        smoke.analyse_profile_data(Path("/profiles/tp8"), expected_ranks=8, timeout=3)


def test_frontend_profiler_is_rejected_before_collection(tmp_path: Path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Torch profiler enabled. AsyncLLM CPU traces will be collected under /profiles\n",
        encoding="utf-8",
    )

    with pytest.raises(smoke.SmokeFailure, match="ignore_frontend=true"):
        smoke.require_worker_only_profiling(server_log)


def _write_server_log(tmp_path: Path, *, expected_keys: int, keys: list[str]):
    server_log = tmp_path / "server.log"
    lines = [
        smoke._STARTUP_REPLAY_CANARY_COMPLETE
        + f" for 2 local SFA layers, {expected_keys} keys "
        + f"({4 * expected_keys} staged graphs).",
        smoke._LIVE_SIGNATURE_VALIDATION,
    ]
    lines.extend(
        smoke._PARITY_PASS
        + f" for key {key}, requests ('request-0',) at sequence lengths "
        + "(4096,) (1/1 live checks, 2 local SFA layers)."
        for key in keys
    )
    server_log.write_text("\n".join(lines), encoding="utf-8")
    return server_log


def test_server_log_accepts_one_parity_check_per_graph_key(tmp_path: Path):
    server_log = _write_server_log(
        tmp_path,
        expected_keys=2,
        keys=["exact-q1-1", "exact-q1-2"],
    )

    assert smoke.check_server_log(server_log) == 2


def test_server_log_rejects_a_graph_key_without_parity(tmp_path: Path):
    server_log = _write_server_log(
        tmp_path,
        expected_keys=2,
        keys=["exact-q1-1"],
    )

    with pytest.raises(
        smoke.SmokeFailure,
        match="parity passed for 1 distinct graph keys; expected 2",
    ):
        smoke.check_server_log(server_log)
