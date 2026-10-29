"""Train and evaluate the DriveMind LSTM on scenario-separated Gold windows.

Run with the main environment (PyTorch, pandas, NumPy, and PyArrow installed):
    python -m etl.prediction.train
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from etl.config import GOLD_DIR, REPO_ROOT
from etl.prediction.lstm_model import TrajectoryLSTM


DEFAULT_CONFIG = REPO_ROOT/"configs"/"trajectory_model.json"
CHECKPOINT = REPO_ROOT/"models"/"trajectory_lstm.pt"


def _local_window(row) -> tuple[np.ndarray, np.ndarray]:
    px, py = np.asarray(row.past_x, dtype=np.float64), np.asarray(row.past_y, dtype=np.float64)
    speed = np.asarray(row.past_speed, dtype=np.float64)
    heading = np.asarray(row.past_heading, dtype=np.float64)
    fx, fy = np.asarray(row.future_x, dtype=np.float64), np.asarray(row.future_y, dtype=np.float64)
    if not (len(px) == len(py) == len(speed) == len(heading)) or not len(px) or len(fx) != len(fy):
        raise ValueError("trajectory window contains inconsistent sequence lengths")
    anchor_x, anchor_y = px[-1], py[-1]
    angle = np.deg2rad(heading[-1])
    c, s = np.cos(angle), np.sin(angle)
    hist_dx, hist_dy = px-anchor_x, py-anchor_y
    future_dx, future_dy = fx-anchor_x, fy-anchor_y
    local_hist = np.column_stack((c*hist_dx+s*hist_dy, -s*hist_dx+c*hist_dy))
    local_future = np.column_stack((c*future_dx+s*future_dy, -s*future_dx+c*future_dy))
    delta_heading = np.deg2rad(heading-heading[-1])
    features = np.column_stack((local_hist, speed, np.sin(delta_heading), np.cos(delta_heading)))
    return features.astype(np.float32), local_future.astype(np.float32)


def load_split(frame: pd.DataFrame, names: list[str], expected_features: int,
               expected_future: int) -> tuple[np.ndarray, np.ndarray]:
    subset = frame[frame.scenario.isin(names)]
    if subset.empty:
        raise ValueError(f"no Gold trajectory windows found for scenarios: {names}")
    x, y = [], []
    for row in subset.itertuples(index=False):
        try:
            features, target = _local_window(row)
            if features.shape[0] == expected_features and target.shape[0] == expected_future \
                    and np.isfinite(features).all() and np.isfinite(target).all():
                x.append(features); y.append(target)
        except (TypeError, ValueError):
            continue
    if not x:
        raise ValueError(f"all windows invalid for scenarios: {names}")
    return np.stack(x), np.stack(y)


def _constant_velocity(x: np.ndarray, future_steps: int) -> np.ndarray:
    # Linear least-squares velocity from the history in the vehicle-local frame.
    pos = x[:, :, :2]
    times = np.arange(pos.shape[1], dtype=np.float64)
    centered = times-times.mean()
    denom = float(centered@centered)
    slope = np.einsum("t,ntc->nc", centered, pos-pos.mean(axis=1, keepdims=True))/denom
    offsets = np.arange(1, future_steps+1, dtype=np.float64)
    return (pos[:, -1:, :]+slope[:, None, :]*offsets[None, :, None]).astype(np.float32)


def _errors(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(prediction-truth, axis=-1)
    return {"ADE_m": float(distance.mean()), "FDE_m": float(distance[:, -1].mean())}


def train(config_path: Path = DEFAULT_CONFIG, gold_path: Path | None = None) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gold_path = gold_path or GOLD_DIR/"carla_telemetry"/"trajectory_windows.parquet"
    frame = pd.read_parquet(gold_path)
    split_names = [config[f"{part}_scenarios"] for part in ("train", "validation", "test")]
    flattened = [name for names in split_names for name in names]
    if len(flattened) != len(set(flattened)):
        raise ValueError("scenario leakage: a scenario appears in more than one split")
    for names in split_names:
        if not set(names).issubset(set(frame.scenario.unique())):
            raise ValueError(f"configured scenario not found in Gold: {set(names)-set(frame.scenario.unique())}")

    input_steps, future_steps = int(config["input_steps"]), int(config["future_steps"])
    x_train, y_train = load_split(frame, config["train_scenarios"], input_steps, future_steps)
    x_val, y_val = load_split(frame, config["validation_scenarios"], input_steps, future_steps)
    x_test, y_test = load_split(frame, config["test_scenarios"], input_steps, future_steps)
    for part, x in (("train",x_train),("validation",x_val),("test",x_test)):
        if len(x) < int(config.get("minimum_samples", 20)):
            raise ValueError(f"insufficient {part} windows: {len(x)}")

    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(config.get("torch_threads", 2))))
    input_mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    input_std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    input_std[input_std < 1e-6] = 1.0
    target_mean = y_train.reshape(-1, 2).mean(axis=0)
    target_std = y_train.reshape(-1, 2).std(axis=0)
    target_std[target_std < 1e-6] = 1.0
    tx = torch.from_numpy((x_train-input_mean)/input_std)
    ty = torch.from_numpy((y_train-target_mean)/target_std)
    vx = torch.from_numpy((x_val-input_mean)/input_std)
    vy = torch.from_numpy((y_val-target_mean)/target_std)
    loader = DataLoader(TensorDataset(tx,ty), batch_size=int(config["batch_size"]), shuffle=True)
    model = TrajectoryLSTM(5, int(config["hidden_size"]), future_steps, int(config["num_layers"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    loss_fn = torch.nn.MSELoss()
    max_epochs, patience = int(config["max_epochs"]), int(config["early_stopping_patience"])
    best_loss, best_state, stale = float("inf"), None, 0
    history: list[dict] = []
    for epoch in range(1,max_epochs+1):
        model.train(); train_losses=[]
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(bx),by); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); train_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_losses=[float(loss_fn(model(bx),by)) for bx,by in DataLoader(TensorDataset(vx,vy),batch_size=128)]
        train_loss=float(np.mean(train_losses)); val_loss=float(np.mean(val_losses))
        history.append({"epoch":epoch,"train_loss":train_loss,"validation_loss":val_loss})
        print(f"epoch {epoch:02d}/{max_epochs} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_loss-1e-6:
            best_loss=val_loss; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=patience: break
    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state); model.eval()

    # Held-out test prediction and per-trajectory inference latency.
    xnorm=(x_test-input_mean)/input_std
    latencies=[]; predictions=[]; cv_latencies=[]; cv_predictions=[]
    with torch.inference_mode():
        for index,sample in enumerate(xnorm):
            begin=time.perf_counter()
            output=model(torch.from_numpy(sample[None,...])).numpy()[0]
            latencies.append((time.perf_counter()-begin)*1000)
            predictions.append(output*target_std+target_mean)
            begin=time.perf_counter()
            cv_predictions.append(_constant_velocity(x_test[index:index+1],future_steps)[0])
            cv_latencies.append((time.perf_counter()-begin)*1000)
    ml_prediction=np.stack(predictions)
    cv_prediction=np.stack(cv_predictions)
    evaluation={
        "model": {**_errors(ml_prediction,y_test),"mean_inference_latency_ms":float(np.mean(latencies)),
                  "p95_inference_latency_ms":float(np.percentile(latencies,95)),"trajectories":len(y_test)},
        "constant_velocity": {**_errors(cv_prediction,y_test),"mean_inference_latency_ms":float(np.mean(cv_latencies)),
                              "trajectories":len(y_test)},
        "test_scenarios":config["test_scenarios"],"coordinate_units":"metres",
        "prediction_horizon_seconds":future_steps/int(config["sample_rate_hz"]),
        "split_method":"scenario-level; no scenario is shared across train, validation, and test",
    }
    checkpoint_payload={
        "model_state_dict":model.state_dict(),
        "config":{k:config[k] for k in ("input_steps","future_steps","sample_rate_hz","input_size","hidden_size","num_layers")},
        "input_mean":input_mean.tolist(),"input_std":input_std.tolist(),
        "target_mean":target_mean.tolist(),"target_std":target_std.tolist(),
        "training_source":f"CARLA 0.9.16 Gold trajectory windows: {gold_path}",
        "train_scenarios":config["train_scenarios"],"validation_scenarios":config["validation_scenarios"],
        "test_scenarios":config["test_scenarios"],"training_samples":len(x_train),
        "validation_samples":len(x_val),"test_samples":len(x_test),"epochs_completed":len(history),
        "final_train_loss":history[-1]["train_loss"],"best_validation_loss":best_loss,
        "evaluation":evaluation,
    }
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint_payload,CHECKPOINT)
    with (CHECKPOINT.parent/"training_history.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    (CHECKPOINT.parent/"evaluation.json").write_text(json.dumps(evaluation,indent=2),encoding="utf-8")
    summary={"checkpoint":str(CHECKPOINT),"training_samples":len(x_train),
             "validation_samples":len(x_val),"test_samples":len(x_test),
             "epochs_completed":len(history),"final_train_loss":history[-1]["train_loss"],
             "best_validation_loss":best_loss,"evaluation":evaluation}
    print(json.dumps(summary,indent=2))
    return summary


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--gold",type=Path,default=None)
    args=parser.parse_args()
    train(args.config,args.gold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
