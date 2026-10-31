/* React UI for the local DriveMind simulation API. */
const h = React.createElement;
const { useEffect, useRef, useState } = React;

class AppBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error) { console.error("DriveMind React render failed:", error); }
  render() {
    if (!this.state.error) return this.props.children;
    return h("main", { className: "render-error" },
      h("h1", null, "Dashboard render error"),
      h("p", null, String(this.state.error?.message || this.state.error)),
      h("p", null, "Restart the local Python dashboard and reload this page."),
      h("button", { className: "button", type: "button", onClick: () => window.location.reload() }, "Reload dashboard"));
  }
}

async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  const contentType = response.headers.get("content-type") || "";
  const result = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function number(value, digits = 1) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—" : Number(value).toFixed(digits);
}

function simClock(value) {
  const seconds = Math.max(0, Number(value) || 0);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}.${Math.floor(seconds * 10) % 10}`;
}

function CanvasScene({ state }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let animationFrame;
    const receivedAt = performance.now();
    function draw() {
    animationFrame = window.requestAnimationFrame(draw);
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = rect.width, height = rect.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0b1521"; ctx.fillRect(0, 0, width, height);
    const roadWidth = Math.min(width * .62, 520);
    const left = (width - roadWidth) / 2, right = left + roadWidth, mid = (left + right) / 2;
    ctx.fillStyle = "#242f3b"; ctx.fillRect(left - 12, 0, roadWidth + 24, height);
    ctx.fillStyle = "#303b47"; ctx.fillRect(left, 0, roadWidth, height);
    ctx.strokeStyle = "#e4bd67"; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(left + 7, 0); ctx.lineTo(left + 7, height);
    ctx.moveTo(right - 7, 0); ctx.lineTo(right - 7, height); ctx.stroke();
    const elapsed = state?.running && !state?.paused
      ? Math.min((performance.now() - receivedAt) / 1000, .6) : 0;
    const position = (state?.ego?.position || 0) + (state?.ego?.speed || 0) * elapsed;
    ctx.strokeStyle = "#8793a0"; ctx.setLineDash([18, 14]);
    ctx.lineDashOffset = -(position * 5 % 32); ctx.beginPath();
    ctx.moveTo(mid, 0); ctx.lineTo(mid, height); ctx.stroke(); ctx.setLineDash([]); ctx.lineDashOffset = 0;
    const egoY = height * .77, scale = height * .68 / 75;
    const laneX = [left + roadWidth * .25, left + roadWidth * .75];
    const actors = state?.actors || [];
    actors.forEach((actor) => {
      const y = egoY - (actor.delta_s || 0) * scale;
      if (y < -25 || y > height + 25) return;
      const x = laneX[actor.lane || 0];
      if (actor.prediction?.length) {
        ctx.strokeStyle = "#d784e4"; ctx.lineWidth = 2; ctx.setLineDash([5, 5]); ctx.beginPath();
        actor.prediction.forEach((point, index) => {
          const px = x + (point[1] - (actor.lane * 3.5)) * scale * .7;
          const py = egoY - (point[0] - (state?.ego?.position || 0)) * scale;
          if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.stroke(); ctx.setLineDash([]);
      }
      if (actor.kind === "obstacle") {
        ctx.fillStyle = "#f4bd58"; ctx.beginPath(); ctx.moveTo(x, y - 10);
        ctx.lineTo(x + 10, y + 9); ctx.lineTo(x - 10, y + 9); ctx.fill(); return;
      }
      roundedRectPath(ctx, x - 12, y - 16, 24, 32, 5);
      ctx.fillStyle = actor.emergency ? "#dc78e8" : actor.kind === "truck" ? "#588dcd" : "#4a7fb7";
      ctx.fill(); ctx.fillStyle = "#bee1ee"; ctx.fillRect(x - 8, y - 10, 16, 6);
      if (actor.emergency) {
        ctx.fillStyle = "#ec8cef"; ctx.fillRect(x - 7, y - 15, 6, 3);
        ctx.fillStyle = "#66d9ff"; ctx.fillRect(x + 1, y - 15, 6, 3);
      }
    });
    const egoX = laneX[state?.ego?.lane || 0];
    roundedRectPath(ctx, egoX - 14, egoY - 19, 28, 38, 6);
    ctx.fillStyle = "#46d9c9"; ctx.fill(); ctx.fillStyle = "#d1fbf5";
    ctx.fillRect(egoX - 9, egoY - 12, 18, 8); ctx.fillStyle = "#c0d0dd";
    ctx.font = "10px monospace"; ctx.fillText("EGO", egoX - 12, egoY + 34);
    ctx.fillStyle = "#8195a8"; ctx.font = "9px monospace";
    ctx.fillText("TOP-DOWN SIMULATION", 12, 18);
    ctx.fillText(`${actors.length} ACTORS`, 12, 34);
    ctx.fillText(`${Math.floor(state?.ego?.position || 0)} m TRAVELLED`, 12, 50);
    }
    draw();
    return () => window.cancelAnimationFrame(animationFrame);
  }, [state]);
  return h("canvas", { ref, className: "scene-canvas", "aria-label": "Top-down driving simulation view" });
}

function roundedRectPath(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, width, height, radius);
  else ctx.rect(x, y, width, height);
}

function HistoryChart({ samples }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
    const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const { width, height } = rect; ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#223349"; ctx.fillStyle = "#8499ac"; ctx.font = "9px monospace";
    for (let index = 0; index < 4; index++) {
      const y = 12 + index * (height - 25) / 3;
      ctx.beginPath(); ctx.moveTo(35, y); ctx.lineTo(width - 8, y); ctx.stroke();
      ctx.fillText(String(100 - index * 33), 3, y + 3);
    }
    if (samples.length < 2) return;
    const plot = (key, color, max) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      samples.forEach((point, index) => {
        const x = 36 + index * (width - 48) / Math.max(1, samples.length - 1);
        const y = height - 12 - Math.min(max, point[key] || 0) / max * (height - 27);
        if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke();
    };
    plot("risk", "#ffc768", 100); plot("speed", "#59aafa", 140);
    ctx.fillStyle = "#ffc768"; ctx.fillText("RISK", width - 74, 10);
    ctx.fillStyle = "#59aafa"; ctx.fillText("KM/H", width - 39, 10);
  }, [samples]);
  return h("canvas", { ref, className: "history-canvas", "aria-label": "Live risk and speed chart" });
}

function CarlaControls({ scenarios, modes, serverExecutable: detectedServer }) {
  const [carla, setCarla] = useState(null);
  const [host, setHost] = useState("127.0.0.1");
  const [serverExecutable, setServerExecutable] = useState(detectedServer || "");
  const [port, setPort] = useState(2000);
  const [scenario, setScenario] = useState("Emergency vehicle approaching");
  const [mode, setMode] = useState("DRIVEMIND");
  const [duration, setDuration] = useState(20);
  const [continuous, setContinuous] = useState(true);
  const [seed, setSeed] = useState(41);
  const [pending, setPending] = useState("");
  const [message, setMessage] = useState("");
  const [cameraTick, setCameraTick] = useState(Date.now());

  async function refresh() {
    try { setCarla(await api("/api/carla/state")); }
    catch (error) { setMessage(error.message); }
  }
  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => { if (detectedServer) setServerExecutable(detectedServer); }, [detectedServer]);
  useEffect(() => {
    if (!carla?.running) return undefined;
    const timer = window.setInterval(() => setCameraTick(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [carla?.running]);

  async function action(path, body) {
    setPending(path); setMessage("");
    try {
      const result = await api(path, body);
      if (path.endsWith("/check")) setMessage(result.message);
      await refresh();
    } catch (error) { setMessage(error.message); await refresh(); }
    finally { setPending(""); }
  }

  const running = !!carla?.running;
  const reachable = !!carla?.server_reachable;
  const progress = carla?.progress;
  const status = carla?.status || "idle";
  return h("section", { className: "panel carla-panel" },
    h("div", { className: "panel-heading" },
      h("div", null, h("p", { className: "eyebrow" }, "LIVE SIMULATOR CONNECTION"),
        h("h2", null, "CARLA control")),
      h("span", { className: `badge ${reachable ? "active" : ""}` }, reachable ? "SERVER CONNECTED" : "SERVER OFFLINE")),
    h("p", { className: "carla-note" },
      "Launch a local CARLA server here or connect to one already running. CARLA RPC normally uses port 2000; the DriveMind web page uses port 8765."),
    h("div", { className: "carla-fields" },
      textField("CARLA SERVER EXE (OPTIONAL)", serverExecutable, setServerExecutable, { type: "text", placeholder: "C:\\CARLA\\WindowsNoEditor\\CarlaUE4.exe", disabled: !!pending }),
      textField("CARLA HOST", host, setHost, { type: "text", disabled: running || !!pending }),
      textField("CARLA RPC PORT (DEFAULT 2000)", port, (value) => setPort(Number(value)), { type: "number", min: 1, max: 65535, disabled: running || !!pending }),
      h("div", { className: "control-group" }, h("label", { htmlFor: "carla-scenario" }, "SCENARIO"),
        h("select", { id: "carla-scenario", value: scenario, onChange: (event) => setScenario(event.target.value), disabled: running || !!pending },
          scenarios.map((item) => h("option", { key: item, value: item }, item)))),
      h("div", { className: "control-group" }, h("label", { htmlFor: "carla-mode" }, "PLANNER"),
        h("select", { id: "carla-mode", value: mode, onChange: (event) => setMode(event.target.value), disabled: running || !!pending },
          modes.map((item) => h("option", { key: item, value: item }, item)))),
      textField("DURATION (SEC)", duration, (value) => setDuration(Number(value)), { type: "number", min: 1, max: 3600, disabled: running || !!pending || continuous }),
      textField("SEED", seed, (value) => setSeed(Number(value)), { type: "number", disabled: running || !!pending })),
    h("label", { className: "carla-continuous" },
      h("input", { type: "checkbox", checked: continuous, disabled: running || !!pending,
        onChange: (event) => setContinuous(event.target.checked) }),
      "Keep running until Stop is pressed"),
    h("div", { className: "carla-actions" },
      h("button", { className: "button", type: "button", disabled: !!pending || !!carla?.server_process_running || !serverExecutable,
        onClick: () => action("/api/carla/server/start", { executable: serverExecutable, port }) }, pending === "/api/carla/server/start" ? "Launching server…" : "Launch CARLA server"),
      h("button", { className: "button danger", type: "button", disabled: !!pending || !carla?.server_process_running || running,
        onClick: () => action("/api/carla/server/stop", {}) }, "Stop CARLA server"),
      h("button", { className: "button", type: "button", disabled: !!pending,
        onClick: () => action("/api/carla/check", { host, port }) }, pending === "/api/carla/check" ? "Checking…" : "Check connection"),
      h("button", { className: "button primary", type: "button", disabled: running || !!pending,
        onClick: () => action("/api/carla/start", { host, port, scenario, mode, duration, seed, continuous }) },
        pending === "/api/carla/start" ? "Starting CARLA…" : "Start CARLA run"),
      h("button", { className: "button danger", type: "button", disabled: !running || !!pending,
        onClick: () => action("/api/carla/stop", {}) }, pending === "/api/carla/stop" ? "Stopping…" : "Stop CARLA run"),
      h("span", { className: `carla-status ${status}` }, status.toUpperCase())),
    message && h("p", { className: "carla-message", role: "status" }, message),
    carla?.error && h("p", { className: "carla-message", role: "alert" }, carla.error),
    progress && h("div", { className: "carla-progress" },
      h("div", { className: "meter-label" }, h("span", null, `${progress.action} · ${number(progress.speed_kmh, 0)} km/h · ${progress.risk_level} risk`),
        h("strong", null, carla.continuous ? `${number(progress.sim_time_s, 1)} s · ${carla.live_available ? (carla.paused ? "PAUSED" : "LIVE") : "LAST UPDATE"}` :
          `${number(progress.sim_time_s, 1)} / ${number(carla.duration_s, 0)} s`)),
      !carla.continuous && h("div", { className: "meter-track" }, h("i", { style: { width: `${progress.progress_pct}%` } }))),
    h("div", { className: "ego-camera-panel" },
      h("div", { className: "carla-output-head" }, h("span", null, "EGO VEHICLE CAMERAS"),
        h("span", null, carla?.camera_available ? `${carla.paused ? "PAUSED" : "LIVE RGB"} · FRONT / REAR` : "NO LIVE CAMERA FEED")),
      carla?.camera_available
        ? h("div", { className: "ego-camera-grid" },
            ["front", "rear"].map((view) => h("figure", { key: view },
              h("figcaption", null, `${view.toUpperCase()} CAMERA`),
              h("img", { src: `/api/carla/camera/${view}.jpg?t=${cameraTick}`,
                alt: `CARLA ego vehicle ${view} camera`, loading: "eager" }))))
        : h("p", { className: "camera-placeholder" }, "Start a CARLA run to see the ego vehicle's live front and rear camera views.")),
    h("div", { className: "carla-output" },
      h("div", { className: "carla-output-head" },
        h("span", null, carla?.log_file || "RUN OUTPUT"),
        carla?.result_line && h("span", null, "RESULTS SAVED")),
      h("pre", null, carla?.log_tail || "Waiting for a CARLA run. Runner output and errors will appear here.")));
}

function textField(label, value, onChange, props = {}) {
  const id = `field-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  return h("div", { className: "control-group", key: label },
    h("label", { htmlFor: id }, label),
    h("input", { id, value, onChange: (event) => onChange(event.target.value), ...props }));
}

function App() {
  const [options, setOptions] = useState({ scenarios: [], modes: [] });
  const [scenario, setScenario] = useState("");
  const [mode, setMode] = useState("DRIVEMIND");
  const [state, setState] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");
  const [pending, setPending] = useState("");
  const [samples, setSamples] = useState([]);
  const stateSnapshot = useRef(null);
  const stateRequest = useRef(0);
  const commandInFlight = useRef(false);

  function rememberState(nextState) {
    stateSnapshot.current = nextState;
    setState(nextState);
    if (nextState?.source === "CARLA") {
      try { window.localStorage.setItem("drivemind-last-carla-state", JSON.stringify(nextState)); }
      catch (_) { /* Storage may be unavailable in private browsing. */ }
    }
  }

  function lastCarlaState() {
    if (stateSnapshot.current?.source === "CARLA") return stateSnapshot.current;
    try {
      const saved = JSON.parse(window.localStorage.getItem("drivemind-last-carla-state") || "null");
      return saved?.source === "CARLA" ? saved : null;
    } catch (_) { return null; }
  }

  async function dashboardSnapshot() {
    const [dashboard, carla] = await Promise.all([
      api("/api/dashboard/state"), api("/api/carla/state"),
    ]);
    if (carla.running && ["running", "stalled"].includes(carla.status)) {
      const lastCarla = carla.live_state || lastCarlaState();
      if (lastCarla) {
        return {
          ...lastCarla,
          running: true,
          paused: !!carla.paused,
          telemetry_stale: carla.status === "stalled" || !carla.live_available,
          notice: carla.status === "stalled"
            ? `${carla.error || "CARLA telemetry is delayed."} Showing the last CARLA frame; Stop remains available.`
            : (!carla.live_available ? "Waiting for the first CARLA telemetry frame." : lastCarla.notice),
        };
      }
    }
    return dashboard;
  }

  async function refresh() {
    if (commandInFlight.current) return;
    const request = ++stateRequest.current;
    try {
      const nextState = await dashboardSnapshot();
      if (request === stateRequest.current && !commandInFlight.current) {
        rememberState(nextState); setConnected(true);
      }
    } catch (_) {
      if (request === stateRequest.current && !commandInFlight.current) setConnected(false);
    }
  }

  useEffect(() => {
    let active = true;
    api("/api/options").then((result) => {
      if (!active) return;
      setOptions(result); setScenario(result.scenarios[0] || "");
      if (result.modes.includes("DRIVEMIND")) setMode("DRIVEMIND");
      return dashboardSnapshot();
    }).then((result) => { if (active && result) { rememberState(result); setConnected(true); } })
      .catch((reason) => { if (active) { setConnected(false); setError(reason.message); } });
    const timer = window.setInterval(refresh, 500);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    const last = state?.last;
    if (!last) return;
    setSamples((previous) => {
      const source = state?.source || "LOCAL";
      const current = previous.at(-1)?.source === source ? previous : [];
      if (current.at(-1)?.time === last.sim_time) return current;
      return [...current, { source, time: last.sim_time, risk: last.risk_score, speed: last.ego_speed * 3.6 }].slice(-60);
    });
  }, [state?.source, state?.last?.sim_time]);

  async function command(path, body = {}) {
    commandInFlight.current = true;
    stateRequest.current += 1;
    setPending(path); setError("");
    try { rememberState(await api(path, body)); await refresh(); }
    catch (reason) { setError(reason.message); }
    finally {
      commandInFlight.current = false;
      setPending("");
      await refresh();
    }
  }

  const running = !!state?.running, paused = !!state?.paused;
  const last = state?.last, metrics = state?.metrics;
  const carlaLive = state?.source === "CARLA";
  const risk = (last?.risk_level || "LOW").toLowerCase();
  const desktopRunning = !!state?.desktop?.running;

  return h(React.Fragment, null,
    h("div", { className: "shell" },
      h("header", { className: "topbar" },
        h("a", { className: "brand", href: "/", "aria-label": "DriveMind home" },
          h("span", { className: "brand-mark" }, "D"),
          h("span", null, "Drive", h("span", { className: "brand-accent" }, "Mind"), h("small", null, "SIMULATION LAB"))),
        h("div", { className: "header-status" }, h("i", { className: connected ? "online" : "" }),
          h("span", null, connected ? "Backend online" : "Backend unavailable"), h("b", null, "LOCAL SESSION"))),
      h("main", null,
        h("section", { className: "intro" },
          h("div", null, h("p", { className: "eyebrow" }, "AUTONOMOUS SYSTEMS / CONTROL ROOM"),
            h("h1", null, "Simulation dashboard"),
            h("p", { className: "subhead" }, carlaLive
              ? "Live CARLA telemetry drives the scene, risk, vehicle controls, and run metrics."
              : "Launch a scenario, inspect live risk, and compare the planner's response.")),
          h("div", { className: "clockbox" }, h("span", null, "SIMULATION TIME"),
            h("strong", null, simClock(last?.sim_time)),
            h("small", null, paused ? "PAUSED" : running ? "RUNNING" : metrics ? "STOPPED" : "READY TO RUN"))),
        h("section", { className: "control-panel panel" },
          h("div", { className: "control-group" }, h("label", { htmlFor: "scenario" }, "SCENARIO"),
            h("select", { id: "scenario", value: scenario, onChange: (event) => setScenario(event.target.value), disabled: running || !!pending },
              options.scenarios.map((item) => h("option", { key: item, value: item }, item)))),
          h("div", { className: "control-group" }, h("label", { htmlFor: "planner" }, "PLANNER MODE"),
            h("select", { id: "planner", value: mode, onChange: (event) => setMode(event.target.value), disabled: running || !!pending },
              options.modes.map((item) => h("option", { key: item, value: item }, item)))),
          h("div", { className: "control-actions" },
            h("button", { className: "button primary", type: "button", disabled: running || !!pending || !connected,
              onClick: () => { setSamples([]); command("/api/start", { scenario, mode, seed: 7 }); } }, h("span", null, "▶"),
              pending === "/api/start" ? " Starting…" : running ? (paused ? " Paused" : " Running") : " Start simulation"),
            h("button", { className: "button", type: "button", disabled: !running || !!pending,
              onClick: () => command(carlaLive ? "/api/carla/pause" : "/api/pause", {}) }, paused ? "▶ Resume" : "Ⅱ Pause"),
            h("button", { className: "button quiet", type: "button", disabled: (carlaLive && running) || !connected || !!pending,
              onClick: () => { setSamples([]); command("/api/reset", { scenario, mode }); } }, "↻ Reset"),
            h("button", { className: "button danger", type: "button", disabled: !running || !!pending,
              onClick: () => command(carlaLive ? "/api/carla/stop" : "/api/stop", {}) },
              pending === "/api/stop" || pending === "/api/carla/stop" ? "■ Stopping…" : "■ Stop"))),
        h("section", { className: "dashboard-grid" },
          h("article", { className: "panel scene-panel" },
            h("div", { className: "panel-heading" }, h("div", null,
              h("p", { className: "eyebrow" }, "LIVE WORLD"), h("h2", null, state?.scenario || "Waiting for simulation")),
              h("span", { className: `badge ${carlaLive || String(state?.prediction_source).includes("LSTM") ? "active" : ""}` },
                carlaLive ? (state?.telemetry_stale ? "CARLA STALE" : "CARLA LIVE")
                  : (state?.prediction_source || "NO MODEL").toUpperCase())),
            h("div", { className: "scene-wrap" }, h(CanvasScene, { state }),
              h("div", { className: "scene-legend" },
                h("span", null, h("i", { className: "legend-ego" }), "Ego vehicle"),
                h("span", null, h("i", { className: "legend-traffic" }), "Traffic"),
                h("span", null, h("i", { className: "legend-emergency" }), "Emergency"),
                h("span", null, h("i", { className: "legend-pred" }), "Predicted path"))),
            h("div", { className: "action-strip" },
              h("span", { className: "action-icon" }, last?.collision ? "!" : last?.brake > .1 ? "↓" : last?.action?.includes("LANE") ? "↗" : "↔"),
              h("div", null, h("small", null, "PLANNER ACTION"), h("strong", null, last?.action || "Start a scenario to see the planner respond")),
              h("span", { className: `risk-pill ${risk}` }, last ? `${last.risk_level} RISK` : "RISK —")),
            h("p", { className: "explanation" }, last?.explanation || "The browser view uses the project's deterministic 2-D simulation. It does not represent camera-based perception.")),
          h("aside", { className: "side-stack" },
            h("article", { className: "panel risk-panel" },
              h("div", { className: "panel-heading compact" }, h("div", null, h("p", { className: "eyebrow" }, "RISK ENGINE"), h("h2", null, "Driving state")),
                h("span", { className: `risk-level ${risk}` }, last?.risk_level || "—")),
              h("div", { className: "risk-meter" }, h("div", { className: "meter-label" }, h("span", null, "Risk score"), h("strong", null, `${number(last?.risk_score, 0)} / 100`)),
                h("div", { className: "meter-track" }, h("i", { className: risk, style: { width: `${Math.max(0, Math.min(100, last?.risk_score || 0))}%` } }))),
              h("div", { className: "metric-list" },
                metricRow("Nearest vehicle", last?.nearest_distance == null ? "—" : `${number(last.nearest_distance)} m`),
                metricRow("Closing speed", last ? `${number(last.relative_speed)} m/s` : "—"),
                metricRow("Time to collision", !last || last.ttc == null || last.ttc === "inf" ? (last ? "∞" : "—") : `${number(last.ttc)} s`),
                metricRow("Emergency status", last?.emergency_status || "—"))),
            h("article", { className: "panel control-state" },
              h("div", { className: "panel-heading compact" }, h("div", null, h("p", { className: "eyebrow" }, "VEHICLE CONTROL"), h("h2", null, "Ego vehicle"))),
              h("div", { className: "vehicle-stats" },
                vehicleStat("SPEED", number(last ? last.ego_speed * 3.6 : null, 0), "km/h"),
                vehicleStat("TARGET", number(last ? last.target_speed * 3.6 : null, 0), "km/h"),
                vehicleStat("LANE", last ? String(last.ego_lane + 1) : "—", last ? (last.ego_lane === 0 ? "left" : "right") : "—")),
              h("div", { className: "control-bars" }, controlBar("Throttle", last?.throttle), controlBar("Brake", last?.brake, true))))),
        h("section", { className: "lower-grid" },
          h("article", { className: "panel chart-panel" }, h("div", { className: "panel-heading compact" },
            h("div", null, h("p", { className: "eyebrow" }, "RUN TELEMETRY"), h("h2", null, "Risk and speed history")),
            h("span", { className: "chart-caption" }, "LIVE · LAST 60 SAMPLES")), h(HistoryChart, { samples })),
          h("article", { className: "panel summary-panel" }, h("div", { className: "panel-heading compact" },
            h("div", null, h("p", { className: "eyebrow" }, "SESSION SUMMARY"), h("h2", null, "Run metrics"))),
            h("div", { className: "summary-grid" },
              summaryStat("Collisions", metrics?.collision_count ?? 0), summaryStat("Near misses", metrics?.near_miss_count ?? 0),
              summaryStat("Minimum TTC", metrics?.minimum_ttc_s == null ? "—" : `${number(metrics.minimum_ttc_s, 2)} s`),
              summaryStat("Average risk", metrics ? number(metrics.average_risk) : "—"),
              summaryStat("Distance", metrics ? `${number(metrics.path_length_m, 0)} m` : "—"),
              summaryStat("Response time", metrics?.response_time_s == null ? "—" : `${number(metrics.response_time_s)} s`)),
            h("p", { className: "file-line" }, "TELEMETRY ", h("span", null, state?.telemetry_file || "No run recorded")))),
        h("section", { className: "panel desktop-panel" },
          h("div", { className: "desktop-copy" }, h("span", { className: "desktop-icon" }, "▣"), h("div", null,
            h("p", { className: "eyebrow" }, "PYTHON DESKTOP VIEW"), h("h2", null, "Open the Tkinter dashboard"),
            h("p", null, "Launch the existing Python dashboard in a separate window, with its planner comparison and model panel."))),
          h("div", { className: "desktop-actions" },
            h("button", { className: "button", type: "button", disabled: desktopRunning || !!pending || !connected,
              onClick: () => command("/api/dashboard/start", {}) }, "Open desktop dashboard"),
            h("button", { className: "button quiet", type: "button", disabled: !desktopRunning || !!pending,
              onClick: () => command("/api/dashboard/stop", {}) }, "Close window")),
          h("span", { className: "desktop-status" }, state?.desktop?.error ? "Launch failed" : desktopRunning ? `Running · PID ${state.desktop.pid}` : "Not running")),
        h(CarlaControls, { scenarios: options.carla_scenarios || [], modes: options.modes,
          serverExecutable: options.carla_server_executable }),
        (error || state?.notice || state?.error) && h("div", { className: "notice show", role: "alert" }, error || state.notice || state.error),
        h("footer", null, h("span", null, "DRIVEMIND · LOCAL SIMULATION VIEW"),
          h("span", null, "Simulation data is illustrative and does not establish real-world safety.")))));
}

function metricRow(label, value) { return h("div", { key: label }, h("span", null, label), h("b", null, value)); }
function vehicleStat(label, value, unit) { return h("div", { key: label }, h("small", null, label), h("strong", null, value), h("em", null, unit)); }
function summaryStat(label, value) { return h("div", { key: label }, h("small", null, label), h("strong", null, value)); }
function controlBar(label, value, isBrake = false) {
  const amount = Math.max(0, Math.min(1, Number(value) || 0));
  return h("div", { key: label }, h("span", null, label),
    h("div", { className: "tiny-track" }, h("i", { style: { width: `${amount * 100}%`, ...(isBrake ? { background: "var(--red)" } : {}) } })),
    h("b", null, amount.toFixed(2)));
}

ReactDOM.createRoot(document.getElementById("root")).render(h(AppBoundary, null, h(App)));
