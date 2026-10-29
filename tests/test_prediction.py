"""Tests for the trained LSTM checkpoint and planner integration."""
from pathlib import Path
import unittest

import numpy as np

from etl.prediction.inference import TrajectoryPredictor
from etl.simulation.core import Actor, DrivingSimulation


CHECKPOINT = Path(__file__).resolve().parents[1]/"models"/"trajectory_lstm.pt"


class _ConflictPredictor:
    status = "LOADED"
    available = True
    input_steps = 20
    sample_rate_hz = 20

    def predict(self, history, *, sample_rate_hz=None):
        target_x = self.sim.ego_s + self.sim.ego_speed*1.5
        xy = np.tile(np.asarray([[target_x, 0.0]], dtype=np.float32), (30,1))
        return {"source":"LSTM", "world_xy":xy, "local_xy":xy, "latency_ms":0.1}


class _BrokenPredictor:
    status="LOADED"
    input_steps=20
    sample_rate_hz=20

    @property
    def available(self):
        return self.status=="LOADED"

    def predict(self, history, *, sample_rate_hz=None):
        raise RuntimeError("test inference failure")


class PredictionTests(unittest.TestCase):
    def test_checkpoint_loads_and_predicts_future_positions(self):
        model = TrajectoryPredictor(CHECKPOINT)
        self.assertEqual(model.status,"LOADED")
        self.assertGreater(model.samples,0)
        self.assertGreater(model.epochs,0)
        history=[(i*0.5,0.0,10.0,0.0) for i in range(20)]
        result=model.predict(history,sample_rate_hz=20)
        self.assertEqual(result["source"],"LSTM")
        self.assertEqual(result["world_xy"].shape,(30,2))
        self.assertTrue(np.isfinite(result["world_xy"]).all())
        self.assertGreater(result["latency_ms"],0)

    def test_missing_checkpoint_reports_fallback_state(self):
        model=TrajectoryPredictor(CHECKPOINT.with_name("missing-model.pt"))
        self.assertEqual(model.status,"NOT TRAINED")
        self.assertFalse(model.available)
        self.assertIsNone(model.predict([(0,0,1,0),(1,0,1,0)]))

    def test_lstm_forecast_changes_risk_aware_planner_action(self):
        predictor=_ConflictPredictor()
        sim=DrivingSimulation("Normal traffic","RISK-AWARE",predictor=predictor)
        predictor.sim=sim
        sim.actors=[Actor("V-01","car",55,16,0)]
        rows=[sim.step(0.1) for _ in range(10)]
        self.assertTrue(sim.model_predictions)
        self.assertEqual(sim.prediction_source,"LSTM")
        intervention=next(row for row in rows if row.action=="CHANGE LANE")
        self.assertIn("LSTM predicts",intervention.explanation)

    def test_inference_error_does_not_stop_simulation(self):
        predictor=_BrokenPredictor()
        sim=DrivingSimulation("Slower vehicle ahead","RISK-AWARE",predictor=predictor)
        for _ in range(10):
            row=sim.step(0.1)
        self.assertEqual(predictor.status,"ERROR")
        self.assertEqual(sim.prediction_source,"CONSTANT VELOCITY")
        self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
