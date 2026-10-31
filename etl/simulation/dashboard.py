"""Dark Tk dashboard for the self-contained DriveMind demonstration."""
from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from etl.config import REPO_ROOT
from etl.simulation.core import (DrivingSimulation, MODES, SCENARIOS,
                                 run_experiments, telemetry_dict)
from etl.simulation.run_demo import write_evaluations

try:
    from etl.prediction.inference import TrajectoryPredictor
except Exception as import_error:  # PyTorch/NumPy unavailable: keep the CV fallback usable.
    _IMPORT_ERROR = str(import_error)
    class TrajectoryPredictor:
        def __init__(self, checkpoint):
            self.checkpoint = Path(checkpoint)
            self.status = "ERROR"
            self.error = f"ML runtime unavailable: {_IMPORT_ERROR}"
            self.samples = self.epochs = 0
            self.train_loss = self.validation_loss = None
            self.evaluation = {}
            self.available = False

BG, PANEL, ROAD, ROAD_EDGE = "#0b1020", "#111a2d", "#303846", "#e5e7eb"
WHITE, MUTED, CYAN, GREEN, AMBER, RED = "#e5edf7", "#8ea0b8", "#45d6d1", "#48d597", "#f4bd58", "#ff657a"


class Dashboard:
    WIDTH, HEIGHT = 1260, 790

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DriveMind | Autonomous Driving Lab")
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.root.minsize(1000, 680)
        self.root.configure(bg=BG)
        self.running, self.paused, self.after_id = True, False, None
        self.scenario_var = tk.StringVar(value=SCENARIOS[0])
        self.mode_var = tk.StringVar(value=MODES[2])
        self.predictor = TrajectoryPredictor(REPO_ROOT/"models"/"trajectory_lstm.pt")
        self.sim = DrivingSimulation(self.scenario_var.get(), self.mode_var.get(), predictor=self.predictor)
        self.experiment_rows: list[dict] = []
        self.log_path: Path | None = None
        self.log_file = None
        self._build()
        self._new_log()
        self.root.bind("<space>", lambda _e: self.toggle_pause())
        for index in range(min(len(SCENARIOS), 9)):
            self.root.bind(f"<Key-{index+1}>", lambda _e, i=index: self.select_scenario(i))
        self._tick()
        self.run_eval()

    def _build(self):
        top = tk.Frame(self.root, bg=BG); top.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(top, text="DRIVEMIND", fg=CYAN, bg=BG, font=("Segoe UI", 21, "bold")).pack(side="left")
        tk.Label(top, text="AUTONOMOUS DRIVING LAB   /   LSTM TRAJECTORY PREDICTION", fg=MUTED, bg=BG,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=18, pady=(7,0))
        controls = tk.Frame(top, bg=BG); controls.pack(side="right")
        self._button(controls, "▶  Start", self.start).pack(side="left", padx=3)
        self._button(controls, "Ⅱ  Pause", self.toggle_pause).pack(side="left", padx=3)
        self._button(controls, "↻  Reset", self.reset).pack(side="left", padx=3)
        body = tk.Frame(self.root, bg=BG); body.pack(fill="both", expand=True, padx=18, pady=8)
        left = tk.Frame(body, bg=BG); left.pack(side="left", fill="both", expand=True)
        selection = tk.Frame(left, bg=BG); selection.pack(fill="x", pady=(0,7))
        tk.Label(selection, text="SCENARIO", fg=MUTED, bg=BG, font=("Segoe UI",9,"bold")).pack(side="left")
        cb = ttk.Combobox(selection, textvariable=self.scenario_var, values=SCENARIOS, state="readonly", width=35)
        cb.pack(side="left", padx=8); cb.bind("<<ComboboxSelected>>", lambda _e: self.reset())
        tk.Label(selection, text="PLANNER", fg=MUTED, bg=BG, font=("Segoe UI",9,"bold")).pack(side="left", padx=(14,0))
        mode = ttk.Combobox(selection, textvariable=self.mode_var, values=MODES, state="readonly", width=15)
        mode.pack(side="left", padx=8); mode.bind("<<ComboboxSelected>>", lambda _e: self.reset())
        self.canvas = tk.Canvas(left, bg="#141c2b", highlightthickness=1, highlightbackground="#29364b")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.draw())
        foot = tk.Frame(left, bg=BG); foot.pack(fill="x", pady=(5,0))
        tk.Label(foot, text=f"Keys  1–{min(len(SCENARIOS), 9)} scenarios   •   Space pause/resume", fg=MUTED, bg=BG,
                 font=("Segoe UI",9)).pack(side="left")
        self.log_label = tk.Label(foot, text="", fg=MUTED, bg=BG, font=("Segoe UI",8))
        self.log_label.pack(side="right")
        right = tk.Frame(body, bg=BG, width=345); right.pack(side="right", fill="y", padx=(12,0)); right.pack_propagate(False)
        self.cards = {}
        for title, key in (("AI PERCEPTION","perception"),("RISK ENGINE","risk"),("VEHICLE CONTROL","control"),("PERFORMANCE","performance")):
            card = tk.Frame(right, bg=PANEL, highlightthickness=1, highlightbackground="#22314a")
            card.pack(fill="x", pady=(0,8))
            tk.Label(card, text=title, fg=CYAN, bg=PANEL, font=("Segoe UI",9,"bold")).pack(anchor="w", padx=11, pady=(8,3))
            label = tk.Label(card, text="", fg=WHITE, bg=PANEL, justify="left", anchor="nw", font=("Consolas",9),
                             wraplength=314)
            label.pack(fill="x", padx=11, pady=(0,9)); self.cards[key] = label
        model_card = tk.Frame(right, bg=PANEL, highlightthickness=1, highlightbackground="#22314a")
        model_card.pack(fill="x", pady=(0,8))
        tk.Label(model_card, text="ML MODEL PANEL", fg=CYAN, bg=PANEL,
                 font=("Segoe UI",9,"bold")).pack(anchor="w", padx=11, pady=(8,3))
        self.ml_label = tk.Label(model_card, text="", fg=WHITE, bg=PANEL, justify="left", anchor="nw",
                                 font=("Consolas",8), wraplength=314)
        self.ml_label.pack(fill="x", padx=11, pady=(0,3))
        self.ml_chart = tk.Canvas(model_card, bg="#0e1626", height=62, highlightthickness=0)
        self.ml_chart.pack(fill="x", padx=9, pady=(0,7))
        exp = tk.Frame(right, bg=PANEL, highlightthickness=1, highlightbackground="#22314a"); exp.pack(fill="both", expand=True)
        header = tk.Frame(exp, bg=PANEL); header.pack(fill="x", padx=10, pady=(8,4))
        tk.Label(header, text="PLANNER COMPARISON", fg=CYAN, bg=PANEL, font=("Segoe UI",9,"bold")).pack(side="left")
        self._button(header, "Run experiments", self.run_eval, small=True).pack(side="right")
        self.chart = tk.Canvas(exp, bg="#0e1626", height=118, highlightthickness=0); self.chart.pack(fill="x", padx=9)
        self.table = tk.Label(exp, text=f"Run experiments for measured results across all {len(SCENARIOS)} scenarios.",
                              fg=MUTED, bg=PANEL, justify="left", anchor="nw", font=("Consolas",8), wraplength=320)
        self.table.pack(fill="both", expand=True, padx=10, pady=8)
        self.status = tk.Label(self.root, text="", fg=MUTED, bg="#080d18", anchor="w", font=("Segoe UI",8))
        self.status.pack(fill="x", side="bottom", ipady=5, padx=8)

    def _button(self, parent, title, callback, small=False):
        return tk.Button(parent, text=title, command=callback, bg="#19253a", fg=WHITE,
                         activebackground="#293b58", activeforeground="white", relief="flat",
                         bd=0, padx=9 if small else 11, pady=4 if small else 6,
                         font=("Segoe UI",8 if small else 9,"bold"), cursor="hand2")

    def _new_log(self):
        folder = REPO_ROOT / "data" / "simulation" / "telemetry"
        folder.mkdir(parents=True, exist_ok=True)
        self.log_path = folder / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        self.log_file = self.log_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.log_file, fieldnames=list(self._telemetry_row(self.sim.step(0)).keys()))
        self.writer.writeheader(); self.log_file.flush()
        # Remove the setup sample from counters and histories.
        self.sim.reset(self.scenario_var.get(), self.mode_var.get())
        self.log_label.configure(text=f"CSV · {self.log_path.name}")

    def reset(self):
        self.running, self.paused = True, False
        if self.log_file: self.log_file.close()
        self.sim = DrivingSimulation(self.scenario_var.get(), self.mode_var.get(), predictor=self.predictor)
        self._new_log(); self.draw()

    def start(self):
        self.paused = False; self.running = True
        if self.after_id is None: self._tick()

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused: self.start()

    def select_scenario(self, index):
        self.scenario_var.set(SCENARIOS[index]); self.reset()

    def _tick(self):
        self.after_id = None
        if self.running and not self.paused:
            row = self.sim.step(0.1)
            self.writer.writerow(self._telemetry_row(row))
            if len(self.sim.telemetry) % 10 == 0: self.log_file.flush()
            self.draw()
        self.after_id = self.root.after(50, self._tick)

    def _telemetry_row(self, row):
        data = telemetry_dict(row)
        data.update({"prediction_source":self.sim.prediction_source,
                     "prediction_latency_ms":self.sim.model_latency_ms,
                     "prediction_horizon_s":self.sim.prediction_horizon_s,
                     "lstm_x_1s":"", "lstm_y_1s":"", "lstm_x_1_5s":"", "lstm_y_1_5s":"",
                     "cv_x_1s":"", "cv_y_1s":""})
        objects=self.sim._perceive()
        nearest=min((o for o in objects if o["lane"]==self.sim.ego_lane and o["s"]>=self.sim.ego_s),
                    key=lambda o:o["s"]-self.sim.ego_s,default=None)
        if nearest:
            actor=next((a for a in self.sim.actors if a.actor_id==nearest["id"]),None)
            if actor:
                data["cv_x_1s"]=actor.s+actor.speed
                data["cv_y_1s"]=actor.lane*3.5
                pred=self.sim.model_predictions.get(actor.actor_id)
                if pred:
                    xy=pred["world_xy"]
                    for key,index in (("1s",min(len(xy)-1,19)),("1_5s",len(xy)-1)):
                        data[f"lstm_x_{key}"]=float(xy[index][0])
                        data[f"lstm_y_{key}"]=float(xy[index][1])
        return data

    def draw(self):
        if not hasattr(self, "canvas"): return
        c = self.canvas; c.delete("all")
        w, h = max(500,c.winfo_width()), max(350,c.winfo_height())
        road_l, road_r = int(w*.20), int(w*.76); mid=(road_l+road_r)//2
        c.create_rectangle(road_l-12,0,road_r+12,h,fill="#252e3b",outline="")
        c.create_rectangle(road_l,0,road_r,h,fill=ROAD,outline=ROAD_EDGE,width=2)
        # Two visible lanes and scrolling dashed markings.
        for y in range(-70,h+70,58):
            yy=int((y + self.sim.time*115) % (h+116)-58)
            c.create_line(mid,yy,mid,yy+31,fill="#a8b1bf",width=3)
        for x in (road_l+6,road_r-6): c.create_line(x,0,x,h,fill="#e4bb5b",width=3)
        ego_y=int(h*.78); scale=min(8.0,h/72)
        # Selected path in highlighted lane.
        lane_x=[road_l+(road_r-road_l)*.25,road_l+(road_r-road_l)*.75]
        ego_x=lane_x[self.sim.ego_lane]
        c.create_line(ego_x,ego_y-170,ego_x,ego_y-22,fill="#3dd6cf",width=3,dash=(5,5),arrow="last")
        for actor in sorted(self.sim.actors,key=lambda a:a.s):
            delta=actor.s-self.sim.ego_s
            y=ego_y-delta*scale
            if y < -40 or y > h+40: continue
            x=lane_x[actor.lane]
            # Constant-velocity baseline at the same 1.5 second horizon as LSTM.
            py=ego_y-(actor.s+actor.speed*self.sim.prediction_horizon_s-self.sim.ego_s)*scale
            c.create_line(x,y, x,py, fill="#8daaf0", dash=(3,4),width=1)
            if actor.model_trajectory:
                pts=[]
                anchor_lane_y=actor.lane*3.5
                for px, lateral in actor.model_trajectory:
                    sx=x+(lateral-anchor_lane_y)*scale*0.8
                    sy=ego_y-(px-self.sim.ego_s)*scale
                    pts.extend((sx,sy))
                if len(pts)>=4:
                    c.create_line(*pts,fill="#e780ff",width=2,dash=(6,3),smooth=True)
                    for index in (19,29):
                        if index < len(actor.model_trajectory):
                            px,lateral=actor.model_trajectory[index]
                            c.create_oval(x+(lateral-anchor_lane_y)*scale*0.8-2,
                                          ego_y-(px-self.sim.ego_s)*scale-2,
                                          x+(lateral-anchor_lane_y)*scale*0.8+2,
                                          ego_y-(px-self.sim.ego_s)*scale+2,
                                          fill="#f7bdff",outline="")
            if actor.kind=="obstacle":
                c.create_polygon(x,y-10,x+10,y+8,x-10,y+8,fill=AMBER,outline="#ffe3a1")
                c.create_text(x,y+19,text="OBSTACLE",fill=AMBER,font=("Segoe UI",7,"bold"))
            else:
                color="#c351df" if actor.emergency else "#4c98ce" if actor.kind=="truck" else "#4775b8"
                c.create_rectangle(x-17,y-27,x+17,y+27,fill=color,outline="#dce6f4",width=1)
                c.create_rectangle(x-12,y-19,x+12,y-7,fill="#b8d8e8",outline="")
                c.create_text(x,y-36,text=actor.actor_id,fill=WHITE,font=("Consolas",8,"bold"))
                if actor.emergency: c.create_text(x,y+36,text="EMERGENCY",fill="#ed86ff",font=("Segoe UI",7,"bold"))
        c.create_rectangle(ego_x-19,ego_y-29,ego_x+19,ego_y+29,fill="#19aa9c",outline="#b5fff1",width=2)
        c.create_rectangle(ego_x-13,ego_y-20,ego_x+13,ego_y-7,fill="#c9f4f2",outline="")
        c.create_text(ego_x,ego_y+40,text="EGO",fill="#69f2db",font=("Segoe UI",9,"bold"))
        c.create_text(road_l+15,18,anchor="w",text="ROAD VIEW  ·  SIMULATION OBJECTS",fill=MUTED,font=("Segoe UI",8,"bold"))
        c.create_line(road_l+18,36,road_l+48,36,fill="#8daaf0",dash=(3,4),width=2)
        c.create_text(road_l+54,36,anchor="w",text="constant velocity",fill=MUTED,font=("Segoe UI",7))
        c.create_line(road_l+160,36,road_l+190,36,fill="#e780ff",dash=(6,3),width=2)
        c.create_text(road_l+196,36,anchor="w",text="LSTM",fill=MUTED,font=("Segoe UI",7))
        c.create_text(road_r-12,18,anchor="e",text=f"t = {self.sim.time:05.1f}s",fill=WHITE,font=("Consolas",9))
        row=self.sim.last
        if not row: return
        c.create_text((road_l+road_r)//2,h-18,text=f"{row.action}     •     {row.risk_level} RISK",fill=RED if row.risk_level=="HIGH" else AMBER if row.risk_level=="MEDIUM" else GREEN,font=("Segoe UI",10,"bold"))
        objects=self.sim._perceive()
        visible=len(objects); nearest="—" if not math.isfinite(row.nearest_distance) else f"{row.nearest_distance:5.1f} m"
        ttc="∞" if not math.isfinite(row.ttc) else f"{row.ttc:4.1f} s"
        active_source=self.sim.prediction_source
        self.cards["perception"].configure(text=f"Objects tracked       {visible:2d}  · stable IDs\nLane detection         SIM MAP  · 2 lanes\nTracking               SIMULATION BASELINE\nTrajectory model       {active_source} · 1.0 / 1.5 s\nEmergency prediction   {row.emergency_status}")
        risk_color=RED if row.risk_level=="HIGH" else AMBER if row.risk_level=="MEDIUM" else GREEN
        self.cards["risk"].configure(text=f"Nearest in lane       {nearest}\nRelative closing      {row.relative_speed:+.1f} m/s\nTime to collision     {ttc}\nRisk score            {row.risk_score:5.1f} / 100\nCategory              {row.risk_level}\nWarnings              {'COLLISION' if row.collision else 'NEAR MISS' if row.near_miss else 'none'}",fg=risk_color)
        self.cards["control"].configure(text=f"Action                {row.action}\nSpeed                 {row.ego_speed*3.6:5.1f} km/h\nTarget                {row.target_speed*3.6:5.1f} km/h\nLane                  {row.ego_lane+1}  ({'left' if row.ego_lane==0 else 'right'})\nSteering              {row.steering:+.2f}\nThrottle / brake      {row.throttle:.2f} / {row.brake:.2f}\nWhy                   {row.explanation}")
        avg=self.sim.risk_total/max(1,self.sim.risk_samples)
        min_ttc="∞" if not math.isfinite(row.min_ttc) else f"{row.min_ttc:.2f}s"
        response = "—" if self.sim.response_time is None else f"{self.sim.response_time:.1f} s"
        self.cards["performance"].configure(text=f"Update rate          20 FPS target\nPlanner latency       {row.latency_ms:.2f} ms\nModules               PERCEPTION · PREDICT\n                      RISK · PLAN · CONTROL\nCollisions / near     {row.collision_count} / {row.near_miss_count}\nMinimum TTC           {min_ttc}\nAverage risk          {avg:.1f} / 100\nResponse time         {response}")
        self._draw_ml_panel(objects)
        self.status.configure(text=f"{self.sim.scenario}  |  {self.sim.mode}  |  prediction: {active_source}  |  telemetry: {self.log_path}")

    def _draw_ml_panel(self, objects):
        p=self.predictor
        eval_data=p.evaluation or {}
        ml=eval_data.get("model",{}); cv=eval_data.get("constant_velocity",{})
        nearest=min((o for o in objects if o["lane"]==self.sim.ego_lane and o["s"]>=self.sim.ego_s),
                    key=lambda o:o["s"]-self.sim.ego_s,default=None)
        prediction=self.sim.model_predictions.get(nearest["id"]) if nearest else None
        coords="warming up — collecting 1.0s of track history"
        if prediction:
            xy=prediction["world_xy"]
            coords=(f"{nearest['id']} @1.0s ({xy[min(19,len(xy)-1),0]:.1f}, {xy[min(19,len(xy)-1),1]:.1f}) m"
                    f"  ·  @1.5s ({xy[-1,0]:.1f}, {xy[-1,1]:.1f}) m")
        loss_train="—" if p.train_loss is None else f"{p.train_loss:.4f}"
        loss_val="—" if p.validation_loss is None else f"{p.validation_loss:.4f}"
        latency=f"{prediction['latency_ms']:.2f}ms" if prediction else "—"
        error=getattr(p,"error","")
        self.ml_label.configure(text=(f"DriveMind LSTM Trajectory Predictor  ·  {p.status}\n"
            f"Type LSTM  |  source CARLA Gold  |  train {p.samples} windows\n"
            f"Epochs {p.epochs}  |  loss train/val {loss_train}/{loss_val}\n"
            f"Checkpoint models/trajectory_lstm.pt  |  infer {latency}\n"
            f"Prediction · 1.0s / 1.5s: {coords}" + (f"\nFallback: {error}" if error else "")))
        self.ml_chart.delete("all")
        if ml and cv:
            width=max(300,self.ml_chart.winfo_width()); base_x=65; row_h=19
            self.ml_chart.create_text(5,8,anchor="nw",text="TEST ADE/FDE (m)  ·  lower is better",fill=MUTED,font=("Segoe UI",7,"bold"))
            max_value=max(float(ml.get("ADE_m",0)),float(cv.get("ADE_m",0)),float(ml.get("FDE_m",0)),float(cv.get("FDE_m",0)),0.1)
            bar_max=width-120
            for i,(metric,label) in enumerate((("ADE_m","ADE"),("FDE_m","FDE"))):
                y=24+i*row_h
                self.ml_chart.create_text(6,y,anchor="nw",text=label,fill=WHITE,font=("Consolas",8,"bold"))
                for j,(name,color) in enumerate((("constant_velocity","#6680a8"),("model","#d77deb"))):
                    val=float(eval_data[name][metric]); segment=bar_max/2-4; x=base_x+j*(segment+12)
                    self.ml_chart.create_rectangle(x,y+2,x+segment*val/max_value,y+12,fill=color,outline="")
                    self.ml_chart.create_text(x+segment*val/max_value+2,y+1,anchor="nw",text=f"{val:.2f}",fill=WHITE,font=("Consolas",7))
            self.ml_chart.create_text(width-92,8,anchor="nw",text="CV / LSTM",fill=WHITE,font=("Segoe UI",7,"bold"))

    def run_eval(self):
        self.status.configure(text=f"Running {len(SCENARIOS)*len(MODES)} comparable experiments "
                                   f"({len(SCENARIOS)} scenarios × {len(MODES)} planners)…")
        self.root.update_idletasks()
        self.experiment_rows = run_experiments()
        path, _json_path = write_evaluations(self.experiment_rows)
        # Display aggregates measured from all matched runs.
        aggregate={m:sum(r["average_risk"] for r in self.experiment_rows if r["mode"]==m)
                   /sum(r["mode"]==m for r in self.experiment_rows) for m in MODES}
        self.chart.delete("all"); self.chart.create_text(8,8,anchor="nw",text=f"Mean risk score across the same {len(SCENARIOS)} scenarios",fill=MUTED,font=("Segoe UI",8))
        colors=["#7c8da6", "#54a7df", CYAN]
        for i,m in enumerate(MODES):
            x=25+i*100; val=aggregate[m]; bh=70*val/100
            self.chart.create_rectangle(x,98-bh,x+42,98,fill=colors[i],outline="")
            self.chart.create_text(x+21,92-bh,text=f"{val:.0f}",fill=WHITE,font=("Consolas",8))
            self.chart.create_text(x+21,108,text=m.replace("-","\n"),fill=MUTED,font=("Segoe UI",7))
        lines=["MODE       COL NEAR  TTC  RISK  RESP", "                      PATH / TIME"]
        for m in MODES:
            rr=[r for r in self.experiment_rows if r["mode"]==m]
            col=sum(r["collision_count"] for r in rr); near=sum(r["near_miss_count"] for r in rr)
            ttcs=[r["minimum_ttc_s"] for r in rr if r["minimum_ttc_s"] is not None]
            ttc=min(ttcs) if ttcs else math.inf
            responses=[r["response_time_s"] for r in rr if r["response_time_s"] is not None]
            resp=f"{sum(responses)/len(responses):.1f}s" if responses else "—"
            path=sum(r["path_length_m"] for r in rr)/len(rr)
            travel=sum(r["travel_time_s"] for r in rr)/len(rr)
            lines.append(f"{m:<10} {col:>2}  {near:>2}  {ttc:>4.1f} {aggregate[m]:>4.0f} {resp:>5}")
            lines.append(f"  avg path {path:>5.0f} m  /  time {travel:>4.0f} s")
        lines.append("Measured simulation output; no safety claim.")
        self.table.configure(text="\n".join(lines))
        self.status.configure(text=f"Experiment report saved: {path}")

    def close(self):
        self.running=False
        if self.after_id: self.root.after_cancel(self.after_id)
        if self.log_file: self.log_file.flush(); self.log_file.close()
        self.root.destroy()


def main():
    root=tk.Tk()
    app=Dashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
