# Robot Curl

MuJoCo experiments for a quadruped that curls its torso into an upright wheel-like shape.

## Project Layout

- `robot_curl/`: active Gymnasium/MuJoCo environment and policy-search helpers
- `scripts/`: training, evaluation, and CEM optimization entrypoints
- `tests/`: lightweight regression tests
- `assets/`: MJCF/XML assets
- `docs/`: design notes and implementation plans
- `legacy/`: older experiments, notebooks, and migration references

## Quick Diagnostics

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m scripts.evaluate_curl --policy zero --episodes 3
python -m scripts.evaluate_curl --policy scripted --episodes 5
python -m scripts.evaluate_curl --policy closed_loop --episodes 5
python -m scripts.evaluate_curl --policy cem --episodes 8
python -m scripts.evaluate_curl --policy feedback_cem --episodes 8
python -m robot_curl.policy_search --limit 80 --episodes 2
python -m robot_curl.policy_search --limit 80 --episodes 2 --closed-loop
python -m scripts.cem_optimize --generations 4 --population 14 --episodes 2 --closed-loop
python -m scripts.cem_optimize --mode feedback --generations 4 --population 14 --episodes 2
```

The `scripted`, `closed_loop`, and `cem` policies are not the final controller. They are reproducible baselines found without PyTorch while PPO dependencies are unavailable. Current CEM baseline:

- stable 250-step episodes
- maximum torso curl around `0.229 rad`
- upright torso (`min_up` near `0.99`)
- no falls over 8 evaluation episodes

A feedback CEM controller is also included. It scales torso curl by the remaining curl error and changes leg phase based on curl progress/contact count. In the current short search it remained stable, but reached only about `0.215 rad`, so it is useful as a diagnostic rather than the best baseline.

PPO experiments can start with:

```powershell
python -m scripts.quick_train --steps 50000
python -m scripts.evaluate_curl --policy model --model quick_runs\curl_smoke\model.zip --norm quick_runs\curl_smoke\vec_normalize.pkl
```

Cloud-oriented SB3 training can start with:

```powershell
python -m scripts.cloud_train --steps 2000000 --envs 32 --device cuda
```

Training artifacts are intentionally ignored by git under `quick_runs/`, `ppo_logs/`, and `ppo_models/`.
