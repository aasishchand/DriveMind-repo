"""Regression coverage for browser controls and malformed API requests."""
import io
import threading
import json
import os
import time
import subprocess
from unittest.mock import Mock

import pytest

from etl import webapp


@pytest.fixture
def service(monkeypatch, tmp_path):
    # The real HTTP smoke check covers the background loop; keep unit transitions deterministic.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(webapp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(webapp.SimulationService, "_new_predictor", lambda self: None)
    instance = webapp.SimulationService()
    yield instance
    instance.stop()


def test_pause_after_stop_does_not_restart_closed_log(service):
    service.start()
    service.stop()
    state = service.pause()
    assert not state["running"]
    assert not state["paused"]
    assert service._log_handle is None


def test_invalid_seed_preserves_running_simulation_and_log(service):
    service.start()
    old_sim, old_log = service.sim, service._log_handle
    with pytest.raises(ValueError):
        service.start(seed="not-an-integer")
    assert service.running
    assert service.sim is old_sim
    assert service._log_handle is old_log
    assert not old_log.closed


def test_pause_resume_and_reset(service):
    service.start(seed=19)
    assert service.pause()["paused"]
    assert not service.pause()["paused"]
    old_log = service._log_handle
    state = service.reset(scenario=webapp.SCENARIOS[-1])
    assert old_log.closed
    assert state["scenario"] == webapp.SCENARIOS[-1]
    assert state["seed"] == 19
    assert state["running"]


@pytest.mark.parametrize("payload", [b"[]", b"null", b'"hello"', b"3"])
def test_api_requires_json_object(payload):
    handler = object.__new__(webapp.Handler)
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)
    with pytest.raises(ValueError, match="JSON object"):
        handler._body()


@pytest.mark.parametrize("length", ["-1", "16385", "invalid"])
def test_api_rejects_invalid_body_length(length):
    handler = object.__new__(webapp.Handler)
    handler.headers = {"Content-Length": length}
    handler.rfile = io.BytesIO(b"{}")
    with pytest.raises(ValueError):
        handler._body()


@pytest.fixture
def carla_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    runner = webapp.CarlaRunner()
    runner.process = Mock()
    runner.process.poll.return_value = None
    monkeypatch.setattr(runner, "server_reachable", lambda *args: True)
    runner.live_state_path.write_text(json.dumps({"source": "CARLA", "running": True}), encoding="utf-8")
    (tmp_path / "front-1.jpg").write_bytes(b"test-frame")
    return runner


@pytest.mark.parametrize("returncode,status", [(0, "completed"), (1, "error")])
def test_finished_carla_cannot_hide_new_local_simulation(carla_runner, returncode, status):
    carla_runner.process.poll.return_value = returncode
    snapshot = carla_runner.state()
    local = {"running": True, "scenario": "Normal traffic"}
    assert snapshot["status"] == status
    assert not snapshot["live_available"]
    assert not snapshot["camera_available"]
    assert webapp.dashboard_state(local, snapshot) == local


def test_disconnected_carla_falls_back_to_local(carla_runner, monkeypatch):
    monkeypatch.setattr(carla_runner, "server_reachable", lambda *args: False)
    snapshot = carla_runner.state()
    assert snapshot["status"] == "disconnected"
    assert snapshot["running"]  # Stop remains usable while the child cleans up.
    assert not snapshot["live_available"]
    assert not snapshot["camera_available"]
    state = webapp.dashboard_state({"running": True}, snapshot)
    assert state["running"]
    assert state.get("source") != "CARLA"
    assert "disconnected" in state["notice"]


def test_stale_carla_frames_are_not_live_but_pause_is_valid(carla_runner):
    old = time.time() - 60
    os.utime(carla_runner.live_state_path, (old, old))
    assert carla_runner.state()["status"] == "stalled"
    assert not carla_runner.state()["camera_available"]
    carla_runner.paused = True
    state = carla_runner.state()
    assert state["live_available"]
    assert webapp.dashboard_state({}, state)["paused"]


def test_live_carla_can_drive_dashboard(carla_runner):
    state = webapp.dashboard_state({"desktop": {"running": False}}, carla_runner.state())
    assert state["source"] == "CARLA"
    assert state["running"]
    assert state["desktop"] == {"running": False}


def test_unresponsive_carla_child_stop_is_bounded():
    process = Mock()
    process.wait.side_effect = [subprocess.TimeoutExpired("carla", 10), 0]
    webapp.CarlaRunner._finish_stop(process)
    process.terminate.assert_called_once()
    assert process.wait.call_count == 2
