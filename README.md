# Robot Curl

MuJoCo experiments for a quadruped that curls its torso into an upright wheel-like shape.

## Quick Diagnostics

From the repository root:

```powershell
python -m pip install -r requirements.txt
python test_mujoco\evaluate_curl.py --policy zero --episodes 3
python test_mujoco\evaluate_curl.py --policy scripted --episodes 5
python test_mujoco\evaluate_curl.py --policy closed_loop --episodes 5
python test_mujoco\evaluate_curl.py --policy cem --episodes 8
python test_mujoco\evaluate_curl.py --policy feedback_cem --episodes 8
python test_mujoco\policy_search.py --limit 80 --episodes 2
python test_mujoco\policy_search.py --limit 80 --episodes 2 --closed-loop
python test_mujoco\cem_optimize.py --generations 4 --population 14 --episodes 2 --closed-loop
python test_mujoco\cem_optimize.py --mode feedback --generations 4 --population 14 --episodes 2
```

The `scripted`, `closed_loop`, and `cem` policies are not the final controller. They are reproducible baselines found without PyTorch while PPO dependencies are unavailable. Current CEM baseline:

- stable 250-step episodes
- maximum torso curl around `0.229 rad`
- upright torso (`min_up` near `0.99`)
- no falls over 8 evaluation episodes

A feedback CEM controller is also included. It scales torso curl by the remaining curl error and changes leg phase based on curl progress/contact count. In the current short search it remained stable, but reached only about `0.215 rad`, so it is useful as a diagnostic rather than the best baseline.

PPO experiments can start with:

```powershell
python test_mujoco\quick_train.py --steps 50000
python test_mujoco\evaluate_curl.py --policy model --model quick_runs\curl_smoke\model.zip --norm quick_runs\curl_smoke\vec_normalize.pkl
```

Training artifacts are intentionally ignored by git under `quick_runs/`, `ppo_logs/`, and `ppo_models/`.
