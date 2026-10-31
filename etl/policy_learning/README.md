# Policy-learning proof of concept

The experience builder gives action examples and collision outcomes one common
CSV schema. It uses the CARLA controller logs already saved under
`data/simulation/telemetry/`:

```powershell
\.venv-carla\Scripts\python.exe -m etl.policy_learning.prepare_experiences
```

The default output is `data/policy_learning/experiences.csv` (local data, not
committed). Rows retain run and scenario IDs, the planner's observed state,
action/control values, camera detections, near-miss labels, and physical
collision outcomes. Collision runs now stop at the first detected contact and
record the other actor type against the last state/action.

## Dataset roles

- **[LEAD](https://github.com/kesai-labs/lead):** expert CARLA driving examples for action supervision. The project
  describes 8,930 routes across 43 scenario types and a full dataset near 1 TB.
  Its 123D Arrow format needs a source adapter before it can be imported here.
- **[CarlAnomaly](https://www.carlanomaly.de/):** anomaly context and collision/autopilot metadata. It describes
  728 CARLA scenarios and a 1.3 TB dataset. Its anomaly labels help characterize
  dangerous situations; they do not provide the corrective driving action.
- **DriveMind CARLA runs:** action and outcome rows in this repository's schema.
  These are the bridge for comparing a baseline action with alternate actions
  in the same seeded situation.

LEAD and CarlAnomaly should be converted into the schema v1 columns in
`prepare_experiences.py` before import. A converted CSV can be appended with
`--normalized-input path\to\converted.csv`. The builder does not download or
copy either full dataset. Roboflow vehicle images remain detector-training
data; they do not contain policy actions or collision outcomes.

## Collision-risk learning loop

CARLA controller telemetry marks the final pre-contact observation with the
collision event and actor type. The builder labels the preceding two seconds as
`collision_context` while retaining each action. These rows train a small
logistic risk predictor; the predictor can request a speed reduction, but does
not learn steering or invent a safe alternate action.

```powershell
\.venv-carla\Scripts\python.exe -m etl.policy_learning.prepare_experiences
\.venv-carla\Scripts\python.exe -m etl.policy_learning.train_risk_model
\.venv-carla\Scripts\python.exe -m etl.policy_learning.evaluate_carla_policy
```

Training requires at least three independent collision runs and three clear
runs. Evaluation compares baseline and candidate over five paired CARLA seeds.
The candidate is promoted to `models/risk_shield.json` only when no episode has
more collisions or near misses, travel distance stays within 20% of baseline,
and aggregate safety or risk improves. Otherwise it remains a candidate. The
simulator loads only this promoted checkpoint by default; pass `--risk-model`
to CARLA to inspect a candidate without promoting it.

This is an offline safety-shield experiment, not unrestricted self-learning.
Evaluation episodes are kept separate by seed, and model weights never change
during a live run. Roboflow images train perception; they do not contain the
actions and outcomes needed for this policy loop.

## Lane-choice behavior cloning

Capture Traffic Manager expert demonstrations with at least four independent
seeds/runs, keeping one complete run for validation:

```powershell
\.venv-carla\Scripts\python.exe -m etl.policy_learning.capture_lane_expert --duration 120 --vehicles 20 --seed 41
\.venv-carla\Scripts\python.exe -m etl.policy_learning.capture_lane_expert --duration 120 --vehicles 20 --seed 42
\.venv-carla\Scripts\python.exe -m etl.policy_learning.capture_lane_expert --duration 120 --vehicles 20 --seed 43
\.venv-carla\Scripts\python.exe -m etl.policy_learning.capture_lane_expert --duration 120 --vehicles 20 --seed 44
```

The recorder saves simulator-state features, expert lane changes, and control
values. The model learns `stay` versus `change adjacent lane`; it does not
regress raw steering angles. Train and evaluate it with:

```powershell
\.venv-carla\Scripts\python.exe -m etl.policy_learning.train_lane_model
\.venv-carla\Scripts\python.exe -m etl.policy_learning.evaluate_lane_policy
```

The trainer holds out one complete capture run. The paired CARLA evaluator
promotes a candidate to `models/lane_policy.json` only if five evaluation seeds
have no collision or near-miss regression, retain at least 80% of baseline
travel distance, and improve aggregate safety or risk. The learned decision
selects the target lane; the waypoint controller steers the vehicle and its gap
check can veto an unsafe change.

## Steering behavior cloning

Steering is trained separately from lane choice. Captures use CARLA Traffic
Manager controls as labels and record lane offset, heading error, lookahead
angle, road curvature, and speed. Collect at least four independent runs, then
fit a candidate with a complete run held out for validation:

```powershell
\.venv-carla\Scripts\python.exe -m etl.policy_learning.capture_steering_expert --duration 120 --vehicles 8 --seed 51
\.venv-carla\Scripts\python.exe -m etl.policy_learning.capture_steering_expert --duration 120 --vehicles 8 --seed 52
\.venv-carla\Scripts\python.exe -m etl.policy_learning.capture_steering_expert --duration 120 --vehicles 8 --seed 53
\.venv-carla\Scripts\python.exe -m etl.policy_learning.capture_steering_expert --duration 120 --vehicles 8 --seed 54
\.venv-carla\Scripts\python.exe -m etl.policy_learning.train_steering_model
```

The trainer writes `models/steering_candidate.json` and reports held-out
MAE/RMSE against a mean-steering baseline. Inspect it in CARLA with
`--steering-model models/steering_candidate.json`; it is not enabled by default.
