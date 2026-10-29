"""Deterministic 2D driving simulation and safety pipeline.

All distances are metres, speeds are m/s, and time is seconds. Perception uses
the simulation's object list as a clearly-labelled baseline (no camera model).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
import time
from collections import deque


LANES = (0, 1)
LANE_NAMES = {0: "left", 1: "right"}
SCENARIOS = ("Normal traffic", "Slower vehicle ahead", "Emergency vehicle approaching", "Obstacle ahead", "Traffic congestion")
MODES = ("BASELINE", "RISK-AWARE", "DRIVEMIND")


@dataclass
class Actor:
    actor_id: str
    kind: str
    s: float
    speed: float
    lane: int
    emergency: bool = False
    width: float = 1.8
    length: float = 4.3
    model_trajectory: list[tuple[float, float]] | None = field(default=None, repr=False)


@dataclass
class Telemetry:
    sim_time: float
    ego_s: float
    ego_speed: float
    ego_lane: int
    target_speed: float
    steering: float
    throttle: float
    brake: float
    action: str
    risk_score: float
    risk_level: str
    nearest_distance: float
    relative_speed: float
    ttc: float
    emergency_status: str
    explanation: str
    latency_ms: float
    collision_count: int
    near_miss_count: int
    min_ttc: float
    collision: bool
    near_miss: bool


def predict_position(s: float, speed: float, horizon: float) -> float:
    """Constant-velocity longitudinal prediction."""
    if horizon < 0:
        raise ValueError("prediction horizon must be non-negative")
    return s + speed * horizon


def calculate_ttc(distance: float, closing_speed: float) -> float:
    """Seconds to collision for positive closing speed; infinity if separating."""
    if distance <= 0:
        return 0.0
    return distance / closing_speed if closing_speed > 1e-6 else math.inf


def assess_risk(distance: float, closing_speed: float, ttc: float,
                obstacle: bool = False, density: float = 0.0,
                lane_change_safe: bool = True, emergency: bool = False) -> tuple[float, str]:
    """Explainable 0-100 risk score; high >=60, medium >=30."""
    distance_risk = max(0.0, min(35.0, (30.0 - distance) / 30.0 * 35.0)) if math.isfinite(distance) else 0.0
    ttc_risk = 35.0 if ttc < 1.5 else (25.0 if ttc < 3 else (12.0 if ttc < 5 else 0.0))
    score = distance_risk + ttc_risk + (20 if obstacle and distance < 25 else 0)
    score += min(10.0, max(0.0, density) * 2.0)
    score += 8.0 if not lane_change_safe else 0.0
    score += max(0.0, 12.0 * (1.0 - distance / 60.0)) if emergency and distance < 60 else 0.0
    score = min(100.0, score)
    return score, "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW"


class DrivingSimulation:
    """Small deterministic world with perception, prediction, planning and control."""

    def __init__(self, scenario: str = SCENARIOS[0], mode: str = "DRIVEMIND", seed: int = 7,
                 predictor=None):
        self.seed = seed
        self.predictor = predictor
        self.reset(scenario, mode)

    def reset(self, scenario: str | None = None, mode: str | None = None) -> None:
        self.scenario = scenario or getattr(self, "scenario", SCENARIOS[0])
        self.mode = mode or getattr(self, "mode", "DRIVEMIND")
        self.rng = random.Random(self.seed)
        self.time = 0.0
        self.ego_s, self.ego_speed, self.ego_lane = 0.0, 18.0, 0
        self.target_speed = 18.0
        self.actors: list[Actor] = []
        if self.scenario == "Normal traffic":
            self.actors = [Actor("V-01", "car", 44, 16, 0), Actor("V-02", "car", 70, 20, 1), Actor("V-03", "truck", 98, 14, 1)]
        elif self.scenario == "Slower vehicle ahead":
            self.actors = [Actor("V-01", "car", 25, 8, 0), Actor("V-02", "car", 60, 19, 1)]
        elif self.scenario == "Emergency vehicle approaching":
            self.actors = [Actor("E-01", "emergency", -36, 27, 0, True),
                           Actor("V-01", "car", 48, 16, 1), Actor("V-02", "car", 82, 19, 0)]
        elif self.scenario == "Obstacle ahead":
            self.actors = [Actor("O-01", "obstacle", 28, 0, 0, width=1.7, length=1.7), Actor("V-01", "car", 53, 17, 1)]
        else:
            self.actors = [Actor(f"V-{i:02d}", "car", 18 + i * 12, 11 + (i % 3), i % 2) for i in range(8)]
        self.collision_count = self.near_miss_count = 0
        self._in_collision = False
        self.min_ttc = math.inf
        self.risk_total = 0.0
        self.response_time: float | None = None
        self.risk_samples = 0
        self.last = None
        self.telemetry: list[Telemetry] = []
        self.actor_histories: dict[str, deque] = {}
        self.model_predictions: dict[str, dict] = {}
        self.model_latency_ms = 0.0
        self.prediction_horizon_s = 1.5

    @property
    def prediction_source(self) -> str:
        return "LSTM" if self.predictor is not None and self.predictor.available \
            and bool(self.model_predictions) else "CONSTANT VELOCITY"

    def _update_model_predictions(self, dt: float) -> None:
        self.model_predictions = {}
        self.model_latency_ms = 0.0
        if self.predictor is None or not self.predictor.available or dt <= 0:
            for actor in self.actors:
                actor.model_trajectory = None
            return
        rate = 1.0/dt
        required_history = max(2, int(math.ceil(rate*self.predictor.input_steps/self.predictor.sample_rate_hz)))
        for actor in self.actors:
            history = self.actor_histories.setdefault(actor.actor_id, deque(maxlen=32))
            history.append((actor.s, actor.lane*3.5, actor.speed, 0.0))
            actor.model_trajectory = None
            if len(history) < required_history:
                continue
            try:
                prediction = self.predictor.predict(list(history), sample_rate_hz=rate)
            except Exception as exc:  # inference errors degrade safely to CV
                self.predictor.status = "ERROR"
                self.predictor.error = f"{type(exc).__name__}: {exc}"
                self.model_predictions.clear()
                for item in self.actors:
                    item.model_trajectory = None
                return
            if prediction:
                actor.model_trajectory = [tuple(map(float, xy)) for xy in prediction["world_xy"]]
                self.model_predictions[actor.actor_id] = prediction
                self.model_latency_ms += float(prediction["latency_ms"])

    def _perceive(self) -> list[dict]:
        # Simulation-aware baseline: perfect actor state and deterministic IDs.
        objects = []
        for actor in self.actors:
            cv_s = predict_position(actor.s, actor.speed, 1.5)
            model_s = actor.model_trajectory[-1][0] if actor.model_trajectory else cv_s
            model_y = actor.model_trajectory[-1][1] if actor.model_trajectory else actor.lane*3.5
            model_lane = 1 if model_y >= 1.75 else 0
            objects.append(dict(id=actor.actor_id, kind=actor.kind,
                                distance=abs(actor.s-self.ego_s), rel_speed=self.ego_speed-actor.speed,
                                speed=actor.speed, lane=actor.lane, emergency=actor.emergency, s=actor.s,
                                predicted_s=actor.s+actor.speed*1.5,
                                ml_predicted_s=model_s, ml_predicted_lane=model_lane,
                                prediction_source="LSTM" if actor.model_trajectory else "CONSTANT VELOCITY"))
        return objects

    def _plan(self, objects: list[dict]) -> tuple[str, float, int, str, float, float, float]:
        relevant = [o for o in objects if o["lane"] == self.ego_lane and o["s"] >= self.ego_s - 3]
        nearest = min(relevant, key=lambda o: max(0, o["s"]-self.ego_s), default=None)
        ahead = max(0.0, nearest["s"] - self.ego_s) if nearest else math.inf
        rel = nearest["rel_speed"] if nearest else 0.0
        ttc = calculate_ttc(ahead, rel)
        obstacle = bool(nearest and nearest["kind"] == "obstacle")
        other_lane = 1 - self.ego_lane
        ego_future = predict_position(self.ego_s, self.ego_speed, 3.0)
        def swept_gap_safe(obj: dict) -> bool:
            if obj["lane"] != other_lane:
                return True
            gap_now = obj["s"] - self.ego_s
            gap_future = predict_position(obj["s"], obj["speed"], 3.0) - ego_future
            # A sign change means the actors cross during the prediction window,
            # even if their final positions appear separated again.
            min_gap = 0.0 if gap_now * gap_future <= 0 else min(abs(gap_now), abs(gap_future))
            return min_gap > 12
        em = [o for o in objects if o["emergency"]]
        approaching = next((o for o in em if o["s"] < self.ego_s and o["rel_speed"] < 0), None)
        em_dist = abs(approaching["s"]-self.ego_s) if approaching else math.inf
        # Keep a clear corridor around emergency traffic after it overtakes too.
        # Normal gap keeping must not send ego back into its lane while nearby.
        emergency_near = next((o for o in em if abs(o["s"]-self.ego_s) < 60), None)
        safe_other = (all(swept_gap_safe(o) for o in objects)
                      and not any(o["emergency"] and o["lane"] == other_lane
                                  and abs(o["s"]-self.ego_s) < 60 for o in objects))
        density = sum(o["distance"] < 30 for o in objects)
        model_conflict = bool(nearest and self.mode != "BASELINE"
                              and nearest["prediction_source"] == "LSTM"
                              and nearest["ml_predicted_lane"] == self.ego_lane
                              and -3.0 <= nearest["ml_predicted_s"]-(self.ego_s+self.ego_speed*1.5) < 10.0)
        risk_ttc = min(ttc, self.prediction_horizon_s) if model_conflict else ttc
        score, level = assess_risk(ahead, rel, risk_ttc, obstacle, density, safe_other, bool(approaching))
        if model_conflict:
            score = min(100.0, score+20.0)
            level = "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW"
        action, target, lane, why = "MAINTAIN SPEED", 18.0, self.ego_lane, "Clear road; hold desired cruise speed."
        if self.mode == "BASELINE":
            if emergency_near and self.ego_lane == emergency_near["lane"]:
                if safe_other:
                    lane, target, action = other_lane, min(self.ego_speed, 12), "YIELD — CLEAR EMERGENCY LANE"
                    why = f"Emergency vehicle nearby ({abs(emergency_near['s']-self.ego_s):.0f} m); keep its lane clear."
                else:
                    target, action = 9.0, "YIELD — SLOW DOWN"
                    why = f"Emergency vehicle nearby ({abs(emergency_near['s']-self.ego_s):.0f} m); lane change is blocked, slow down."
            elif ahead < 12:
                action, target, why = "BRAKE", 0.0, "Reference planner brakes at a short fixed distance."
            elif nearest and ahead < 30 and rel > 0:
                action, target, why = "SLOW DOWN", min(18.0, max(6.0, nearest["speed"])), "Reference gap keeping."
        else:
            if emergency_near and self.ego_lane == emergency_near["lane"]:
                if safe_other:
                    lane, target, action = other_lane, min(self.ego_speed, 12), "YIELD — CLEAR EMERGENCY LANE"
                    why = f"Emergency vehicle nearby ({abs(emergency_near['s']-self.ego_s):.0f} m); keep its lane clear."
                else:
                    target, action = 9.0, "YIELD — SLOW DOWN"
                    why = f"Emergency vehicle nearby ({abs(emergency_near['s']-self.ego_s):.0f} m); lane change is blocked, slow down."
            elif nearest and (ahead < 38 and rel > 0 or model_conflict):
                if safe_other and (ahead > 12 or model_conflict):
                    lane, action, target = 1-self.ego_lane, "CHANGE LANE", 18.0
                    why = ("LSTM predicts a close future approach; adjacent lane is clear."
                           if model_conflict else f"Adjacent lane is clear; pass slower {nearest['kind']} safely.")
                else:
                    target = max(0, min(self.ego_speed, nearest.get("speed", 0) + max(0, ahead-8)*0.35))
                    action = "BRAKE" if model_conflict or ttc < 2.2 or ahead < 12 or obstacle else "SLOW DOWN"
                    why = ("LSTM forecast reduces the safe following speed." if model_conflict
                           else f"Following distance {ahead:.1f} m, TTC {ttc:.1f} s; preserve a safe gap.")
            elif level == "HIGH":
                action, target = "BRAKE", 5.0
                why = f"High predicted risk ({score:.0f}/100); reduce speed."
        return action, target, lane, why, ahead, rel, ttc

    def step(self, dt: float = 0.1) -> Telemetry:
        start = time.perf_counter()
        self._update_model_predictions(dt)
        objects = self._perceive()
        action, desired, target_lane, why, distance, rel, ttc = self._plan(objects)
        self.target_speed = desired
        if target_lane != self.ego_lane:
            self.ego_lane = target_lane
        # Functional longitudinal controller: bounded acceleration and braking.
        delta = self.target_speed - self.ego_speed
        acceleration = max(-7.0, min(2.8, delta * 1.8))
        self.ego_speed = max(0, self.ego_speed + acceleration * dt)
        throttle = max(0, min(1, acceleration / 2.8))
        brake = max(0, min(1, -acceleration / 7.0))
        for actor in self.actors:
            actor.s += actor.speed * dt
        self.ego_s += self.ego_speed * dt
        # Remove actors well behind the camera and add deterministic traffic ahead.
        self.actors = [a for a in self.actors if a.s > self.ego_s-55]
        if len([a for a in self.actors if a.kind == "car"]) < 4 and self.scenario in ("Normal traffic", "Traffic congestion"):
            lane = self.rng.choice(LANES)
            self.actors.append(Actor(f"V-{int(self.time*10)+10:02d}", "car", self.ego_s+75+self.rng.random()*35, 13+self.rng.random()*7, lane))
        contacts = [a for a in self.actors if a.lane == self.ego_lane and abs(a.s-self.ego_s) < (3.0 if a.kind != "obstacle" else 2.4)]
        collision = bool(contacts)
        if collision and not self._in_collision:
            self.collision_count += 1
        self._in_collision = collision
        if collision:
            self.ego_speed = min(self.ego_speed, 1.0)
            why = "COLLISION detected; controller applies emergency braking."
        near = distance < 8 or ttc < 1.5
        if near and not (self.last and self.last.near_miss):
            self.near_miss_count += 1
        self.min_ttc = min(self.min_ttc, ttc)
        risk, level = assess_risk(distance, rel, ttc,
                                  any(a.kind == "obstacle" and abs(a.s-self.ego_s)<25 for a in self.actors),
                                  sum(abs(a.s-self.ego_s)<30 for a in self.actors), True,
                                  any(a.emergency and a.s < self.ego_s+60 for a in self.actors))
        if self.mode != "BASELINE" and any(
            item["prediction_source"] == "LSTM" and item["ml_predicted_lane"] == self.ego_lane
            and -3.0 <= item["ml_predicted_s"]-(self.ego_s+self.ego_speed*self.prediction_horizon_s) < 10.0
            for item in objects
        ):
            risk = min(100.0, risk+20.0)
            level = "HIGH" if risk >= 60 else "MEDIUM" if risk >= 30 else "LOW"
        elapsed = (time.perf_counter()-start)*1000
        elapsed += self.model_latency_ms
        em = next((a for a in self.actors if a.emergency), None)
        em_status = f"APPROACHING · {abs(em.s-self.ego_s):.0f} m" if em and em.s < self.ego_s else ("PRESENT" if em else "NONE")
        steering = (self.ego_lane-0.5)*0.55
        self.time += dt
        record = Telemetry(self.time, self.ego_s, self.ego_speed, self.ego_lane, self.target_speed,
                           steering, throttle, brake, action, risk, level, distance, rel, ttc, em_status,
                           why, elapsed, self.collision_count, self.near_miss_count,
                           self.min_ttc, collision, near)
        self.last = record
        self.telemetry.append(record)
        self.risk_total += risk
        self.risk_samples += 1
        if self.response_time is None and action != "MAINTAIN SPEED":
            self.response_time = self.time
        return record

    def metrics(self) -> dict:
        return {"mode": self.mode, "scenario": self.scenario,
                "collision_count": self.collision_count, "near_miss_count": self.near_miss_count,
                "minimum_ttc_s": self.min_ttc if math.isfinite(self.min_ttc) else None,
                "average_risk": self.risk_total/max(1,self.risk_samples),
                "response_time_s": self.response_time,
                "travel_time_s": self.time, "path_length_m": self.ego_s,
                "ticks": len(self.telemetry)}


def run_experiments(duration: float = 25.0) -> list[dict]:
    """Run each planner on identical scenario seeds and return measured metrics."""
    out = []
    for scenario in SCENARIOS:
        for mode in MODES:
            sim = DrivingSimulation(scenario, mode, seed=41)
            for _ in range(int(duration/0.1)):
                sim.step(0.1)
            out.append(sim.metrics())
    return out


def telemetry_dict(row: Telemetry) -> dict:
    data = asdict(row)
    for key in ("ttc", "min_ttc"):
        if not math.isfinite(data[key]):
            data[key] = "inf"
    return data
