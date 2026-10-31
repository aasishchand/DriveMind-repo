"""Pure standard-library checks for the live simulation safety pipeline."""
import math
import unittest

from etl.simulation.core import (Actor, DrivingSimulation, assess_risk, calculate_ttc,
                                 predict_position, run_experiments, SCENARIOS, MODES)


class SimulationCoreTests(unittest.TestCase):
    def test_ttc_closing_and_separating(self):
        self.assertEqual(calculate_ttc(20, 5), 4)
        self.assertTrue(math.isinf(calculate_ttc(20, -2)))
        self.assertEqual(calculate_ttc(0, -2), 0)

    def test_constant_velocity_prediction(self):
        self.assertEqual(predict_position(10, 4, 2.5), 20)
        with self.assertRaises(ValueError):
            predict_position(10, 4, -1)

    def test_risk_categories_and_obstacle(self):
        self.assertEqual(assess_risk(100, 0, math.inf)[1], "LOW")
        self.assertEqual(assess_risk(5, 8, 0.6)[1], "HIGH")
        self.assertGreater(assess_risk(10, 0, math.inf, obstacle=True)[0],
                           assess_risk(10, 0, math.inf)[0])

    def test_planner_controller_changes_speed_and_lane(self):
        sim = DrivingSimulation("Slower vehicle ahead", "RISK-AWARE")
        initial_speed = sim.ego_speed
        row = sim.step()
        self.assertEqual(row.action, "CHANGE LANE")
        self.assertEqual(sim.ego_lane, 1)
        # When both lanes are blocked, the longitudinal controller brakes.
        sim = DrivingSimulation("Normal traffic", "RISK-AWARE")
        sim.actors = [Actor("blocked-ahead", "car", 12, 8, 0),
                      Actor("blocked-adjacent", "car", 8, 8, 1)]
        initial_speed = sim.ego_speed
        row = sim.step()
        self.assertIn(row.action, {"SLOW DOWN", "BRAKE"})
        self.assertLess(sim.ego_speed, initial_speed)
        emergency = DrivingSimulation("Emergency vehicle approaching", "DRIVEMIND")
        row = emergency.step()
        self.assertIn("YIELD", row.action)
        self.assertEqual(emergency.ego_lane, 1)
        for _ in range(249):
            emergency.step()
        self.assertEqual(emergency.collision_count, 0)
        self.assertEqual(emergency.near_miss_count, 0)
        reference = DrivingSimulation("Emergency vehicle approaching", "BASELINE")
        self.assertIn("YIELD", reference.step().action)
        for _ in range(249):
            reference.step()
        self.assertEqual(reference.collision_count, 0)
        obstacle = DrivingSimulation("Obstacle ahead", "RISK-AWARE")
        obstacle.step()
        self.assertEqual(obstacle.ego_lane, 1)


    def test_experiment_grid_is_comparable_and_measured(self):
        rows = run_experiments(duration=0.5)
        self.assertEqual(len(rows), len(SCENARIOS) * len(MODES))
        self.assertEqual({(row["scenario"], row["mode"]) for row in rows},
                         {(scenario, mode) for scenario in SCENARIOS for mode in MODES})
        self.assertEqual({row["mode"] for row in rows}, {"BASELINE", "RISK-AWARE", "DRIVEMIND"})
        self.assertTrue(all(row["ticks"] == 5 for row in rows))

    def test_learned_risk_does_not_weaken_imminent_braking(self):
        class HighRiskModel:
            threshold = 0.5

            def probability(self, features):
                return 1.0

        sim = DrivingSimulation("Normal traffic", "RISK-AWARE",
                                risk_model=HighRiskModel(), lane_model=False)
        sim.actors = [Actor("ahead", "car", 9, 0, 0),
                      Actor("adjacent", "car", 8, 0, 1)]
        sim.risk_probability_ema = 1.0
        sim.risk_shield_active = True
        before = sim.ego_speed
        row = sim.step(0.1)
        self.assertEqual(row.action, "BRAKE")
        self.assertAlmostEqual(before - sim.ego_speed, 0.7)

    def test_matched_speed_is_not_reported_as_active_braking(self):
        sim = DrivingSimulation("Normal traffic", "DRIVEMIND", risk_model=False, lane_model=False)
        sim.ego_speed = 14.0 + 1e-12
        sim.actors = [Actor("ahead", "car", 8.0, 14.0, 0),
                      Actor("adjacent", "car", 8.0, 14.0, 1)]
        row = sim.step()
        self.assertEqual(row.action, "FOLLOW VEHICLE")
        self.assertAlmostEqual(row.brake, 0.0)


if __name__ == "__main__":
    unittest.main()
