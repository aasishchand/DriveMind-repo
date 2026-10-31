"""Local browser dashboard and simulation API for DriveMind.

Run with ``python -m etl.webapp`` and open http://127.0.0.1:8765.
Only the local machine can access the server by default.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from etl.config import REPO_ROOT
from etl.simulation.core import DrivingSimulation, MODES, SCENARIOS, telemetry_dict

WEB_DIR = REPO_ROOT / "web"
HOST, PORT = "127.0.0.1", 8765
CARLA_CAMERA_SCENARIOS = (
    "Approaching ambulance", "Police vehicle crossing intersection", "Fire truck ahead",
    "Ordinary vehicle negative control", "Partially occluded emergency vehicle",
    "Low-confidence false-positive case",
)
CARLA_SCENARIOS = tuple(dict.fromkeys((*SCENARIOS, *CARLA_CAMERA_SCENARIOS)))
CARLA_PYTHON = REPO_ROOT / ".venv-carla" / "Scripts" / "python.exe"
CARLA_SERVER_CANDIDATES = (
    Path.home() / "Downloads" / "CARLA_0.9.16" / "CarlaUE4.exe",
    Path.home() / "CARLA_0.9.16" / "CarlaUE4.exe",
)


def find_carla_server():
    return next((path for path in CARLA_SERVER_CANDIDATES if path.is_file()), None)


def _finite(value):
    """Convert non-finite floats to JSON nulls."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


class SimulationService:
    """Thread-safe owner of the simulation, telemetry log, and Tk dashboard."""

    def __init__(self):
        self.lock = threading.RLock()
        self.sim: DrivingSimulation | None = None
        self.running = False
        self.paused = False
        self.scenario = SCENARIOS[0]
        self.mode = MODES[2]
        self.seed = 7
        self.error = None
        self.log_path: Path | None = None
        self._log_handle = None
        self._writer = None
        self._dashboard: subprocess.Popen | None = None
        self._dashboard_log_path: Path | None = None
        self._dashboard_error: str | None = None
        self._thread = threading.Thread(target=self._loop, name="drivemind-sim", daemon=True)
        self._thread.start()

    def _new_predictor(self):
        try:
            from etl.prediction.inference import TrajectoryPredictor
            return TrajectoryPredictor(REPO_ROOT / "models" / "trajectory_lstm.pt")
        except Exception:
            return None

    def start(self, scenario=SCENARIOS[0], mode=MODES[2], seed=7):
        if scenario not in SCENARIOS:
            raise ValueError("Unknown simulation scenario")
        if mode not in MODES:
            raise ValueError("Unknown planner mode")
        seed = int(seed)
        with self.lock:
            simulation = DrivingSimulation(scenario, mode, seed=seed,
                                           predictor=self._new_predictor())
            self.running, self.paused = False, False
            self._close_log()
            self.scenario, self.mode, self.seed = scenario, mode, seed
            self.error = None
            self.sim = simulation
            self._open_log()
            self.running, self.paused = True, False
        return self.state()

    def reset(self, scenario=None, mode=None):
        return self.start(scenario or self.scenario, mode or self.mode, self.seed)

    def pause(self):
        with self.lock:
            if self.sim is None or not self.running:
                return self.state()
            self.paused = not self.paused
            self.running = True
            return self.state()

    def stop(self):
        with self.lock:
            self.running, self.paused = False, False
            self._close_log()
            return self.state()

    def _open_log(self):
        folder = REPO_ROOT / "data" / "simulation" / "telemetry"
        folder.mkdir(parents=True, exist_ok=True)
        self.log_path = folder / f"web_run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        self._log_handle = self.log_path.open("w", newline="", encoding="utf-8")
        self._writer = None

    def _close_log(self):
        if self._log_handle:
            self._log_handle.flush()
            self._log_handle.close()
        self._log_handle, self._writer = None, None

    def _loop(self):
        while True:
            started = time.perf_counter()
            with self.lock:
                if self.running and not self.paused and self.sim is not None:
                    try:
                        row = self.sim.step(0.1)
                        values = telemetry_dict(row)
                        if self._writer is None:
                            self._writer = csv.DictWriter(self._log_handle,
                                                         fieldnames=list(values.keys()))
                            self._writer.writeheader()
                        self._writer.writerow(values)
                        if len(self.sim.telemetry) % 10 == 0:
                            self._log_handle.flush()
                    except Exception as exc:  # keep the API alive and surface the simulation error
                        self.error = f"{type(exc).__name__}: {exc}"
                        self.running = False
                        self._close_log()
            time.sleep(max(0.0, 0.05 - (time.perf_counter() - started)))

    def _actor_state(self, actor):
        delta = actor.s - self.sim.ego_s
        predicted = []
        if actor.model_trajectory:
            predicted = [[float(s), float(lane_y)] for s, lane_y in actor.model_trajectory]
        return {"id": actor.actor_id, "kind": actor.kind, "lane": actor.lane,
                "emergency": bool(actor.emergency), "delta_s": float(delta),
                "speed": float(actor.speed), "prediction": predicted}

    def state(self):
        with self.lock:
            sim = self.sim
            last = telemetry_dict(sim.last) if sim and sim.last else None
            return _finite({
                "running": self.running,
                "paused": self.paused,
                "scenario": self.scenario,
                "mode": self.mode,
                "seed": self.seed,
                "error": self.error,
                "telemetry_file": str(self.log_path.relative_to(REPO_ROOT)) if self.log_path else None,
                "desktop": {"running": bool(self._dashboard and self._dashboard.poll() is None),
                            "pid": self._dashboard.pid if self._dashboard and self._dashboard.poll() is None else None,
                            "log_file": str(self._dashboard_log_path.relative_to(REPO_ROOT))
                                if self._dashboard_log_path else None,
                            "error": self._dashboard_error},
                "last": last,
                "metrics": sim.metrics() if sim else None,
                "actors": [self._actor_state(actor) for actor in sim.actors] if sim else [],
                "ego": ({"speed": sim.ego_speed, "lane": sim.ego_lane,
                         "position": sim.ego_s} if sim else None),
                "prediction_source": sim.prediction_source if sim else "Not started",
            })

    def launch_tk_dashboard(self):
        with self.lock:
            if self._dashboard and self._dashboard.poll() is None:
                return {"running": True, "pid": self._dashboard.pid,
                        "log_file": str(self._dashboard_log_path.relative_to(REPO_ROOT))
                            if self._dashboard_log_path else None}
            folder = REPO_ROOT / "data" / "simulation" / "telemetry"
            folder.mkdir(parents=True, exist_ok=True)
            log_path = folder / f"tk_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
            python_path = Path(sys.executable)
            gui_python = python_path.with_name("pythonw.exe") if sys.platform == "win32" else python_path
            if not gui_python.is_file():
                gui_python = python_path
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform == "win32" else 0
            try:
                with log_path.open("w", encoding="utf-8", errors="replace") as log:
                    process = subprocess.Popen([str(gui_python), str(REPO_ROOT / "main.py")],
                                               cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=flags)
            except OSError as exc:
                self._dashboard_error = f"Could not launch Tk dashboard: {exc}"
                raise RuntimeError(self._dashboard_error) from exc
            self._dashboard, self._dashboard_log_path = process, log_path
            self._dashboard_error = None
        time.sleep(0.4)
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            self._dashboard_error = tail.strip() or f"Tk dashboard exited with code {process.returncode}."
            raise RuntimeError(f"Tk dashboard failed to start. Log: {log_path}\n{self._dashboard_error}")
        return {"running": True, "pid": process.pid,
                "log_file": str(log_path.relative_to(REPO_ROOT))}

    def close_tk_dashboard(self):
        with self.lock:
            if not self._dashboard or self._dashboard.poll() is not None:
                return {"running": False, "error": self._dashboard_error}
            self._dashboard.terminate()
            try:
                self._dashboard.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._dashboard.kill()
            self._dashboard = None
            return {"running": False, "error": None}


class CarlaRunner:
    """Manage a DriveMind CARLA client run and optionally its local server."""

    PROGRESS_RE = re.compile(
        r"\[carla_demo\] t=\s*([\d.]+)s action=(.*?) speed=\s*([\d.]+)km/h "
        r"risk=([A-Z]+):([\d.]+) collision_events=(\d+)"
    )
    # CARLA can pause for map streaming or an occasional slow sensor/inference tick.
    # Keep the last observed scene for a short gap instead of switching dashboards.
    LIVE_STATE_MAX_AGE_SECONDS = 30.0

    def __init__(self):
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.python_path: Path | None = None
        self.host, self.port = "127.0.0.1", 2000
        self.scenario, self.mode = "Emergency vehicle approaching", "DRIVEMIND"
        self.duration, self.seed = 20.0, 41
        self.continuous = False
        self.log_path: Path | None = None
        self.stop_path: Path | None = None
        self.pause_path: Path | None = None
        self.paused = False
        self.started_at: float | None = None
        self.stop_requested = False
        self.start_error: str | None = None
        self.server_process: subprocess.Popen | None = None
        self.server_executable: Path | None = None
        self.server_log_path: Path | None = None
        self.frame_dir = Path(tempfile.mkdtemp(prefix=f"drivemind_carla_frames_{os.getpid()}_"))
        self.live_state_path = self.frame_dir / "live-state.json"

    def live_state(self):
        try:
            with self.live_state_path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, json.JSONDecodeError):
            return None

    def camera_frame(self, view: str):
        if view not in ("front", "rear"):
            return None
        for _ in range(2):
            try:
                candidates = sorted(self.frame_dir.glob(f"{view}-*.jpg"),
                                    key=lambda path: path.stat().st_mtime, reverse=True)
                return candidates[0].read_bytes() if candidates else None
            except (FileNotFoundError, OSError):
                continue
        return None

    @staticmethod
    def server_reachable(host: str, port: int, timeout: float = 0.35) -> bool:
        if int(port) == PORT and str(host).strip().lower() in ("127.0.0.1", "localhost", "::1"):
            # The browser dashboard's HTTP listener is not a CARLA RPC server.
            return False
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    @staticmethod
    def resolve_python() -> Path:
        configured = os.environ.get("DRIVEMIND_CARLA_PYTHON")
        candidate = Path(configured) if configured else CARLA_PYTHON
        if not candidate.is_file():
            candidate = Path(sys.executable)
        return candidate

    def check(self, host="127.0.0.1", port=2000):
        host = str(host).strip()
        port = int(port)
        if not host or not 1 <= port <= 65535:
            raise ValueError("Enter a valid CARLA host and port (1–65535).")
        if port == PORT and host.lower() in ("127.0.0.1", "localhost", "::1"):
            return {"host": host, "port": port, "reachable": False,
                    "message": "Port 8765 is the DriveMind web page. CARLA RPC normally uses port 2000."}
        reachable = self.server_reachable(host, port)
        return {"host": host, "port": port, "reachable": reachable,
                "message": (f"CARLA server reachable at {host}:{port}." if reachable else
                            f"Cannot reach CARLA at {host}:{port}. Start CarlaUE4 first.")}

    def start_server(self, executable, port=2000):
        executable = Path(str(executable)).expanduser().resolve()
        port = int(port)
        if not executable.is_file() or executable.name.lower() != "carlaue4.exe":
            raise ValueError("Select the packaged CarlaUE4.exe executable.")
        if not 1 <= port <= 65535:
            raise ValueError("Enter a valid CARLA port (1–65535).")
        if port == PORT:
            raise ValueError("Port 8765 is reserved for the DriveMind web page. Use CARLA RPC port 2000.")
        with self.lock:
            if self.server_process and self.server_process.poll() is None:
                raise ValueError("A CARLA server launched by this app is already running.")
        if self.server_reachable("127.0.0.1", port):
            raise ValueError(f"A server already responds on 127.0.0.1:{port}; connect to it instead.")
        folder = REPO_ROOT / "data" / "simulation" / "telemetry"
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / f"carla_server_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform == "win32" else 0
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen([str(executable), f"-carla-rpc-port={port}"],
                                       cwd=executable.parent, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        with self.lock:
            self.server_process, self.server_executable, self.server_log_path = process, executable, log_path
            self.host, self.port = "127.0.0.1", port
        return self.state()

    def stop_server(self):
        with self.lock:
            process = self.server_process
            if not process or process.poll() is not None:
                return self.state()
            process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
        return self.state()

    def start(self, scenario, mode, duration, seed, host="127.0.0.1", port=2000,
              continuous=False):
        host, port = str(host).strip(), int(port)
        duration, seed = float(duration), int(seed)
        continuous = bool(continuous)
        if scenario not in CARLA_SCENARIOS:
            raise ValueError("Unknown CARLA scenario")
        if mode not in MODES:
            raise ValueError("Unknown planner mode")
        if not host or not 1 <= port <= 65535:
            raise ValueError("Enter a valid CARLA host and port (1–65535).")
        if port == PORT and host.lower() in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Port 8765 is the DriveMind web page. Set CARLA RPC port to 2000.")
        if not continuous and not 1 <= duration <= 3600:
            raise ValueError("Duration must be between 1 and 3600 seconds.")
        if not -2_147_483_648 <= seed <= 2_147_483_647:
            raise ValueError("Seed is outside the supported integer range.")
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("A CARLA run is already active.")
        if not self.server_reachable(host, port):
            raise ConnectionError(f"Cannot reach CARLA at {host}:{port}. Start CarlaUE4 first.")

        python_path = self.resolve_python()
        try:
            probe = subprocess.run([str(python_path), "-c", "import carla"], cwd=REPO_ROOT,
                                   capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Could not check the CARLA Python environment: {exc}") from exc
        if probe.returncode:
            detail = (probe.stderr or probe.stdout).strip().splitlines()
            hint = detail[-1] if detail else "CARLA Python package is missing"
            raise RuntimeError(f"{python_path} cannot import carla: {hint}")

        folder = REPO_ROOT / "data" / "simulation" / "telemetry"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.live_state_path.unlink(missing_ok=True)
        log_path = folder / f"carla_web_{stamp}.log"
        stop_path = Path(tempfile.gettempdir()) / f"drivemind_carla_stop_{os.getpid()}.flag"
        pause_path = Path(tempfile.gettempdir()) / f"drivemind_carla_pause_{os.getpid()}.flag"
        stop_path.unlink(missing_ok=True)
        pause_path.unlink(missing_ok=True)
        command = [str(python_path), "-u", "-m", "etl.simulation.carla_demo",
                   "--host", host, "--port", str(port),
                   "--scenario", scenario, "--mode", mode, "--seed", str(seed),
                   "--no-camera-preview", "--stop-file", str(stop_path), "--pause-file", str(pause_path),
                   "--web-frame-dir", str(self.frame_dir)]
        command.extend(["--continuous"] if continuous else ["--duration", str(duration)])
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform == "win32" else 0
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, creationflags=flags)
        except OSError as exc:
            raise RuntimeError(f"Could not launch CARLA runner: {exc}") from exc
        with self.lock:
            self.process, self.python_path = process, python_path
            self.host, self.port, self.scenario, self.mode = host, port, scenario, mode
            self.duration, self.seed, self.continuous = (None if continuous else duration), seed, continuous
            self.log_path, self.stop_path, self.pause_path = log_path, stop_path, pause_path
            self.started_at, self.stop_requested, self.paused, self.start_error = time.time(), False, False, None
        return self.state()

    def toggle_pause(self):
        with self.lock:
            process, pause_path = self.process, self.pause_path
            if not process or process.poll() is not None or not pause_path:
                return self.state()
            self.paused = not self.paused
            if self.paused:
                pause_path.touch(exist_ok=True)
            else:
                pause_path.unlink(missing_ok=True)
            return self.state()

    def stop(self):
        with self.lock:
            process, stop_path = self.process, self.stop_path
            if not process or process.poll() is not None:
                return self.state()
            first_request = not self.stop_requested
            self.stop_requested = True
            if stop_path:
                stop_path.touch(exist_ok=True)
            if first_request:
                threading.Thread(target=self._finish_stop, args=(process,),
                                 name="drivemind-carla-stop", daemon=True).start()
            return self.state()

    @staticmethod
    def _finish_stop(process):
        """Bound cleanup when a disconnected CARLA client cannot finish its RPC calls."""
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            except OSError:
                pass  # The child may exit between poll/wait and terminate.

    def _tail(self, limit=16_000):
        if not self.log_path or not self.log_path.is_file():
            return ""
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - limit))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def state(self):
        with self.lock:
            process = self.process
            returncode = process.poll() if process else None
            if self.start_error:
                status = "error"
            elif not process:
                status = "idle"
            elif returncode is None:
                status = "stopping" if self.stop_requested else "running"
            elif self.stop_requested:
                status = "stopped"
            elif returncode == 0:
                status = "completed"
            else:
                status = "error"
            reachable = self.server_reachable(self.host, self.port)
            live = self.live_state()
            try:
                live_age = max(0.0, time.time() - self.live_state_path.stat().st_mtime)
            except OSError:
                live_age = None
            fresh = live_age is not None and live_age <= self.LIVE_STATE_MAX_AGE_SECONDS
            error = self.start_error
            if status == "running" and not reachable:
                status = "disconnected"
                error = "CARLA disconnected. Stop this run, restart the CARLA server, then start a new run."
            elif status == "running" and live and not fresh and not self.paused:
                status = "stalled"
                error = "CARLA telemetry stopped updating. Stop the run and check the simulator."
            live_available = status == "running" and isinstance(live, dict) and (fresh or self.paused)
            tail = self._tail()
            progress = None
            for line in reversed(tail.splitlines()):
                match = self.PROGRESS_RE.search(line)
                if match:
                    elapsed = float(match.group(1))
                    progress = {"sim_time_s": elapsed, "action": match.group(2).strip(),
                                "speed_kmh": float(match.group(3)), "risk_level": match.group(4),
                                "risk_score": float(match.group(5)),
                                "collision_events": int(match.group(6)),
                                "progress_pct": (None if self.continuous else
                                                 min(100.0, 100.0 * elapsed / self.duration))}
                    break
            result_line = next((line for line in reversed(tail.splitlines())
                                if "[carla_demo] results:" in line), None)
            log_rel = str(self.log_path.relative_to(REPO_ROOT)) if self.log_path else None
            server_process = self.server_process
            server_running = bool(server_process and server_process.poll() is None)
            server_log = str(self.server_log_path.relative_to(REPO_ROOT)) if self.server_log_path else None
            return {"status": status, "running": bool(process and returncode is None),
                    "server_reachable": reachable,
                    "live_available": live_available, "live_age_s": live_age,
                    "server_process_running": server_running,
                    "server_executable": str(self.server_executable) if self.server_executable else None,
                    "server_log_file": server_log,
                    "camera_available": live_available and any(next(self.frame_dir.glob(f"{view}-*.jpg"), None)
                                             is not None for view in ("front", "rear")),
                    "host": self.host, "port": self.port, "scenario": self.scenario,
                    "mode": self.mode, "duration_s": self.duration, "seed": self.seed,
                    "continuous": self.continuous,
                    "paused": self.paused,
                    "python": str(self.python_path) if self.python_path else str(self.resolve_python()),
                    "log_file": log_rel, "progress": progress,
                    "live_state": live if isinstance(live, dict) and status in ("running", "stalled") else None,
                    "result_line": result_line,
                    "log_tail": tail[-6000:], "error": error}

    def shutdown(self):
        self.stop()
        process = self.process
        if process and process.poll() is None:
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.terminate()
        self.stop_server()


SERVICE = SimulationService()
CARLA = CarlaRunner()


def dashboard_state(local_state, carla_state):
    """Keep the CARLA controls with an active run, even during a telemetry pause."""
    status = carla_state.get("status")
    live_available = carla_state.get("live_available")
    if (status == "running" and live_available) or (
        status == "stalled" and carla_state.get("running") and carla_state.get("live_state")
    ):
        live = dict(carla_state["live_state"])
        live["running"] = bool(carla_state.get("running"))
        live["paused"] = bool(carla_state.get("paused"))
        live["desktop"] = local_state.get("desktop")
        live["telemetry_stale"] = status == "stalled"
        if status == "stalled":
            live["notice"] = (carla_state.get("error") or "CARLA telemetry is delayed.") + \
                " Showing the last CARLA frame; Stop remains available."
        return live
    local = dict(local_state)
    if status == "disconnected":
        local["notice"] = carla_state.get("error") + " Showing the local 2-D simulation."
    return local


class DriveMindHTTPServer(ThreadingHTTPServer):
    """Ignore clients that close an HTTP connection before a response completes."""

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "DriveMindLocal/1.0"

    def log_message(self, fmt, *args):
        print(f"[drivemind-web] {self.address_string()} {fmt % args}")

    def _json(self, payload, status=200):
        data = json.dumps(_finite(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= 16_384:
            raise ValueError("Request body must be between 0 and 16384 bytes")
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        return body

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/carla/camera/") and path.endswith(".jpg"):
            view = path.rsplit("/", 1)[-1][:-4]
            frame = CARLA.camera_frame(view)
            if frame is None:
                return self._json({"error": "Camera frame is not available yet. Start a CARLA run."}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            try:
                self.wfile.write(frame)
            except (ConnectionResetError, BrokenPipeError):
                pass
            return
        if path == "/api/state":
            return self._json(SERVICE.state())
        if path == "/api/carla/state":
            return self._json(CARLA.state())
        if path == "/api/dashboard/state":
            local_state = SERVICE.state()
            carla_state = CARLA.state()
            return self._json(dashboard_state(local_state, carla_state))
        if path == "/api/options":
            return self._json({"scenarios": SCENARIOS, "modes": MODES,
                               "carla_scenarios": CARLA_SCENARIOS,
                               "carla_server_executable": str(find_carla_server() or "")})
        if path == "/" or path == "/index.html":
            path = "/index.html"
        if path.startswith("/api/"):
            return self._json({"error": "Not found"}, 404)
        file_path = (WEB_DIR / unquote(path.lstrip("/"))).resolve()
        if WEB_DIR.resolve() not in file_path.parents and file_path != WEB_DIR.resolve():
            return self._json({"error": "Not found"}, 404)
        if not file_path.is_file():
            return self._json({"error": "Not found"}, 404)
        data = file_path.read_bytes()
        self.send_response(200)
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
        }.get(file_path.suffix.lower(), mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/start":
                body = self._body()
                return self._json(SERVICE.start(body.get("scenario", SCENARIOS[0]),
                                                body.get("mode", MODES[2]),
                                                body.get("seed", 7)))
            if path == "/api/pause":
                return self._json(SERVICE.pause())
            if path == "/api/reset":
                body = self._body()
                return self._json(SERVICE.reset(body.get("scenario"), body.get("mode")))
            if path == "/api/stop":
                return self._json(SERVICE.stop())
            if path == "/api/dashboard/start":
                return self._json(SERVICE.launch_tk_dashboard())
            if path == "/api/dashboard/stop":
                return self._json(SERVICE.close_tk_dashboard())
            if path == "/api/carla/check":
                body = self._body()
                return self._json(CARLA.check(body.get("host", "127.0.0.1"),
                                              body.get("port", 2000)))
            if path == "/api/carla/start":
                body = self._body()
                return self._json(CARLA.start(
                    body.get("scenario", "Emergency vehicle approaching"),
                    body.get("mode", "DRIVEMIND"), body.get("duration", 20),
                    body.get("seed", 41), body.get("host", "127.0.0.1"),
                    body.get("port", 2000), body.get("continuous", False)))
            if path == "/api/carla/stop":
                return self._json(CARLA.stop())
            if path == "/api/carla/pause":
                return self._json(CARLA.toggle_pause())
            if path == "/api/carla/server/start":
                body = self._body()
                return self._json(CARLA.start_server(body.get("executable", ""), body.get("port", 2000)))
            if path == "/api/carla/server/stop":
                return self._json(CARLA.stop_server())
            return self._json({"error": "Not found"}, 404)
        except ConnectionError as exc:
            return self._json({"error": str(exc)}, 503)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    server = DriveMindHTTPServer((HOST, PORT), Handler)
    print(f"DriveMind browser dashboard: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping DriveMind web dashboard.")
    finally:
        SERVICE.stop()
        SERVICE.close_tk_dashboard()
        CARLA.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
