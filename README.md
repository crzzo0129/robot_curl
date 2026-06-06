# Robot Curl

MuJoCo experiments for a quadruped that curls its torso into an upright wheel-like shape.

## Quick Diagnostics

From the repository root:

```powershell
python -m pip install -r requirements.txt
python test_mujoco\evaluate_curl.py --policy zero --episodes 3
python test_mujoco\evaluate_curl.py --policy scripted --episodes 5
python test_mujoco\policy_search.py --limit 80 --episodes 2
```

The `scripted` policy is not the final controller. It is a reproducible open-loop baseline found by grid search. Current baseline target:

- stable 250-step episodes
- maximum torso curl around `0.22 rad`
- upright torso (`min_up` near `0.99`)

PPO experiments can start with:

```powershell
python test_mujoco\quick_train.py --steps 50000
python test_mujoco\evaluate_curl.py --policy model --model quick_runs\curl_smoke\model.zip --norm quick_runs\curl_smoke\vec_normalize.pkl
```

Training artifacts are intentionally ignored by git under `quick_runs/`, `ppo_logs/`, and `ppo_models/`.
