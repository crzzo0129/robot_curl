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

Enable Weights & Biases logging with `--wandb`:

```powershell
wandb login
python -m scripts.quick_train --steps 50000 --wandb --wandb-project robot-curl --wandb-name curl-smoke
```

For cloud runs:

```bash
python -m scripts.cloud_train --steps 2000000 --envs 32 --device cuda --curl-goal 0.20 --wandb --wandb-project robot-curl --wandb-name cloud-curl-020
```

Use `--wandb-mode offline` if the machine cannot reach W&B during training, then sync the run later with `wandb sync`.

Task settings are shared by training and evaluation. For example, start with an easier curl target:

```powershell
python -m scripts.quick_train --steps 50000 --curl-goal 0.20
python -m scripts.evaluate_curl --policy model --model quick_runs\curl_smoke\model.zip --norm quick_runs\curl_smoke\vec_normalize.pkl --curl-goal 0.20
```

Useful task knobs:

- `--curl-goal`: target torso curl in radians
- `--max-episode-steps`: episode length at 50 Hz control rate
- `--action-scale`: per-step joint target increment limit
- `--reward-curl`, `--reward-progress`, `--reward-contact`, `--reward-upright`: main reward weights

Cloud-oriented SB3 training can start with:

```powershell
python -m scripts.cloud_train --steps 2000000 --envs 32 --device cuda --curl-goal 0.20
```

## MJX Backend Smoke Test

The MJX backend is separate from the Gymnasium environment while parity is being checked. On the cloud machine:

```bash
conda activate mjx312
python -m robot_curl_mjx.smoke_test --steps 1 --curl-goal 0.20
```

The first `mjx.step` can spend tens of seconds in XLA compilation and may print GPU autotuning warnings. This has been observed on CUDA 12.8 with `jax-cuda12-*==0.6.2`; the important line is `jax_backend=gpu`.

Expected output includes staged progress lines (`stage=init_env`, `stage=reset_start`, `stage=step_start`) plus the JAX backend, observation shape, action size, final reward, and termination flags. The smoke test defaults to `--action-repeat 1 --settle-steps 0` so first-run XLA compilation is easier to diagnose. If it works, increase `--steps`; if it is still slow, use `TF_CPP_MIN_LOG_LEVEL=2` to hide XLA warning noise. Add `--skip-reward` or `--skip-terminated` to isolate whether reward/contact checks are the slow part. This smoke test only verifies MJX model loading and stepping; MJX PPO training is the next layer.

## MJX PPO Training

The MJX training entrypoint uses Brax PPO with many parallel MJX environments. Start with a small cloud smoke run:

```bash
conda activate mjx312
python -m scripts.mjx_train --steps 10000 --envs 128 --episode-length 128 --curl-goal 0.20
```

The first run will compile JAX/XLA kernels, so the first progress output can be slow. Successful runs save Brax parameters under `mjx_runs/curl_smoke/params`. Enable W&B logging with:

```bash
python -m scripts.mjx_train --steps 200000 --envs 512 --episode-length 128 --curl-goal 0.20 --wandb --wandb-project robot-curl --wandb-name mjx-curl-020
```

The training entrypoint follows the same shape as the Pupper cloud notebook: it sets cloud-friendly XLA/MuJoCo defaults, uses an explicit policy network (`--hidden-layers 256 128 128 128 --activation elu` by default), and logs evaluation reward/length through Brax PPO. Training never renders videos at intermediate evaluations. After PPO finishes, it renders one deterministic `final_policy.mp4` at `320x240` by default and uploads it as `final_policy_video` to the same W&B run. Use `--no-final-policy-video` for diagnostic runs that should skip this final render.

Evaluate a saved MJX policy and render a CPU/OSMesa video:

```bash
MUJOCO_GL=osmesa python -m scripts.mjx_playback --params mjx_runs/curl_smoke/params --episodes 3 --video mjx_runs/curl_smoke/playback.mp4
```

Playback prints per-episode `total_reward`, `max_curl`, `min_upright`, contact count, and done status. It also writes `mjx_runs/curl_smoke/eval.csv`. Use `--no-video` when you only need metrics.

Training artifacts are intentionally ignored by git under `quick_runs/`, `ppo_logs/`, and `ppo_models/`.
