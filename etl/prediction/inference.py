"""Load a trained LSTM checkpoint and infer a tracked vehicle's path."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from etl.prediction.lstm_model import TrajectoryLSTM


class TrajectoryPredictor:
    MODEL_NAME = "DriveMind LSTM Trajectory Predictor"

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint)
        self.status = "NOT TRAINED"
        self.source = "unavailable"
        self.samples = 0
        self.epochs = 0
        self.train_loss = None
        self.validation_loss = None
        self.evaluation = {}
        self.input_steps = 20
        self.future_steps = 30
        self.sample_rate_hz = 20
        self.input_mean = self.input_std = self.target_mean = self.target_std = None
        self.model = None
        self.last_latency_ms = 0.0
        try:
            if not self.checkpoint.is_file():
                return
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            config = payload["config"]
            self.input_steps = int(config["input_steps"])
            self.future_steps = int(config["future_steps"])
            self.sample_rate_hz = int(config["sample_rate_hz"])
            self.source = str(payload["training_source"])
            self.samples = int(payload["training_samples"])
            self.epochs = int(payload["epochs_completed"])
            self.train_loss = float(payload["final_train_loss"])
            self.validation_loss = float(payload["best_validation_loss"])
            self.evaluation = payload.get("evaluation", {})
            self.input_mean = np.asarray(payload["input_mean"], dtype=np.float32)
            self.input_std = np.asarray(payload["input_std"], dtype=np.float32)
            self.target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
            self.target_std = np.asarray(payload["target_std"], dtype=np.float32)
            self.model = TrajectoryLSTM(
                input_size=int(config["input_size"]),
                hidden_size=int(config["hidden_size"]),
                future_steps=self.future_steps,
                num_layers=int(config["num_layers"]),
            )
            self.model.load_state_dict(payload["model_state_dict"])
            self.model.eval()
            self.status = "LOADED"
        except Exception as exc:  # corrupted/mismatched model must use CV fallback
            self.status = "ERROR"
            self.error = f"{type(exc).__name__}: {exc}"
            self.model = None

    @property
    def available(self) -> bool:
        return self.status == "LOADED" and self.model is not None

    def predict(self, history: list[tuple[float, float, float, float]],
                *, sample_rate_hz: float | None = None) -> dict | None:
        """Predict world XY from history tuples (x, y, speed, heading degrees)."""
        if not self.available or len(history) < 2:
            return None
        start = time.perf_counter()
        raw = np.asarray(history, dtype=np.float64)
        source_rate = float(sample_rate_hz or self.sample_rate_hz)
        if len(raw) != self.input_steps or abs(source_rate-self.sample_rate_hz) > 1e-6:
            old_times = np.arange(len(raw), dtype=np.float64) / source_rate
            end = old_times[-1]
            new_times = np.linspace(max(0, end-(self.input_steps-1)/self.sample_rate_hz),
                                    end, self.input_steps)
            headings = np.unwrap(np.deg2rad(raw[:, 3]))
            values = [np.interp(new_times, old_times, raw[:, col]) for col in range(3)]
            values.append(np.rad2deg(np.interp(new_times, old_times, headings)))
            raw = np.column_stack(values)

        anchor_x, anchor_y = raw[-1, 0], raw[-1, 1]
        anchor_heading = np.deg2rad(raw[-1, 3])
        cos_h, sin_h = np.cos(anchor_heading), np.sin(anchor_heading)
        dx, dy = raw[:, 0]-anchor_x, raw[:, 1]-anchor_y
        local_x = cos_h*dx + sin_h*dy
        local_y = -sin_h*dx + cos_h*dy
        delta_heading = np.deg2rad(raw[:, 3]-raw[-1, 3])
        features = np.column_stack((local_x, local_y, raw[:, 2],
                                    np.sin(delta_heading), np.cos(delta_heading))).astype(np.float32)
        standardized = (features-self.input_mean)/self.input_std
        with torch.inference_mode():
            tensor = torch.from_numpy(standardized[None, ...])
            normalized = self.model(tensor).numpy()[0]
        local = normalized*self.target_std+self.target_mean
        world_dx = cos_h*local[:, 0] - sin_h*local[:, 1]
        world_dy = sin_h*local[:, 0] + cos_h*local[:, 1]
        self.last_latency_ms = (time.perf_counter()-start)*1000
        return {"source": "LSTM", "world_xy": np.column_stack((anchor_x+world_dx, anchor_y+world_dy)),
                "local_xy": local, "latency_ms": self.last_latency_ms}
