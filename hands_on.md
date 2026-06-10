# Robot Curl Hands-On State

This document is a handoff note for another AI agent or developer joining the
project. It describes what the project is trying to do, what currently works,
what is experimental, and where to continue.

## Project Goal

The project explores a quadruped robot that can curl its torso into an upright
wheel-like shape, then potentially roll or transition back to walking. The near
term goal is not side-rolling and not a full front flip yet. The desired first
behavior is:

- stay upright, not side-lying
- curl into a vertical tire-like posture
- learn to move the legs/feet to make torso curling feasible, instead of simply
  forcing a target torso angle through joint stiffness
- later study how the number of torso segments/free joints affects rolling
  smoothness, roughly like a polygon approximating a circle

## Repository State

Repository root on the local machine:

```text
C:\Users\12481\Desktop\OH-WorkSpace\robot_description
```

GitHub remote:

```text
https://github.com/crzzo0129/robot_curl.git
```

Most recent local commit at the time of this note:

```text
3db57b6 Consolidate MJX training pipeline with WandB videos
```

Important: the MJX pipeline/W&B policy-video work described below is already in
the current local commit. Push status may still need checking from git/GitHub.
Check local state with:

```bash
git status --short
```

Files touched by that pipeline work:

```text
README.md
robot_curl/wandb_utils.py
robot_curl_mjx/pipeline.py
scripts/mjx_playback.py
scripts/mjx_train.py
tests/test_mjx_train_entry.py
```

## Main Code Layout

```text
assets/quadruped.xml
```

Current MJCF robot model. It has 13 controlled joints:

- 1 torso hinge: `torso_hinge`
- 12 leg joints: abduction/flex/knee for four legs

The XML uses position actuators. The MJX training path currently drives those
position actuators directly.

```text
robot_curl/
```

Original Gymnasium/MuJoCo environment and non-MJX helpers.

- `env.py`: `QuadrupedFoldEnv`, `CurlTaskConfig`, joint names, reward logic
- `config_args.py`: shared CLI task args
- `wandb_utils.py`: shared W&B helpers
- `policy_search.py`: CEM/scripted baseline search

```text
robot_curl_mjx/
```

MJX-specific work.

- `env.py`: small stateful MJX smoke-test env
- `brax_env.py`: Brax-compatible functional MJX env for PPO
- `smoke_test.py`: diagnostic MJX reset/step smoke test
- `pipeline.py`: in-progress shared pipeline helpers for cloud runtime,
  network factory, and W&B video callbacks

```text
scripts/
```

Entrypoints.

- `quick_train.py`: short SB3 PPO training on local/Gym env
- `cloud_train.py`: SB3 PPO cloud-oriented entry
- `evaluate_curl.py`: evaluate SB3/model/scripted/CEM policies
- `cem_optimize.py`: CEM baseline optimization
- `mjx_train.py`: MJX/Brax PPO training entry
- `mjx_playback.py`: load MJX params, evaluate episodes, render OSMesa video

```text
tests/
```

Lightweight regression tests. Many tests are intentionally import/CLI-level so
they can run without initializing JAX/GPU locally.

## Proven Cloud Environment

The user has a working cloud conda env:

```text
conda env: mjx312
Python: 3.12.13
GPU: NVIDIA GPU, JAX backend = gpu
CUDA: 12.8
jax                  0.6.2
jaxlib               0.6.2
jax-cuda12-plugin    0.6.2
jax-cuda12-pjrt      0.6.2
flax                 0.10.6
brax                 0.14.2
mujoco               3.9.0
mujoco-mjx           3.9.0
pyopengl             3.1.10
```

Cloud rendering is selected before MuJoCo is imported. On headless Linux,
`MUJOCO_GL=auto` selects EGL without initializing GLFW/X11. To force
CPU/OSMesa manually, use:

```bash
MUJOCO_GL=osmesa
```

The user prefers running real MJX/Brax training tests on cloud, not locally.
Local checks should be limited to syntax, imports, and CLI parsing unless the
local environment has matching JAX/Brax/MuJoCo packages.

## What Has Worked

### MJX Smoke Test

Command used on cloud:

```bash
python -m robot_curl_mjx.smoke_test --steps 1 --curl-goal 0.20
```

Observed successful output included:

```text
jax_backend=gpu
obs_shape=(37,) action_size=13
steps=1 reward=0.250 terminated=False truncated=False
```

The first `mjx.step` may pause for XLA compilation and print warnings like:

```text
All configs were filtered out because none of them sufficiently match the hints
```

This warning has been seen on the user's CUDA 12.8/JAX 0.6.2 setup and is not
by itself a failure.

### MJX PPO Training

Command run by the user:

```bash
python -m scripts.mjx_train --steps 10000 --envs 128 --episode-length 128 --curl-goal 0.20
```

This completed successfully after the Brax metrics pytree bug was fixed.

Observed result:

```text
stage=train_start steps=10000 envs=128 episode_length=128 action_repeat=1
steps=0 eval_reward=-333.473
steps=10240 eval_reward=10.739
steps=20480 eval_reward=21.422
steps=30720 eval_reward=30.619
steps=40960 eval_reward=13.228
stage=train_done saved=mjx_runs/curl_smoke/params
training/sps ~= 14917
eval/avg_episode_length ~= 127.48
eval/episode_reward_std ~= 124.66
```

Interpretation:

- the MJX PPO entry can train and save params
- the policy starts learning something, but it is not stable yet
- high eval reward variance means behavior is inconsistent
- episode length near 128 suggests it is usually not immediately falling

### Brax Metrics Bug Already Fixed

Earlier error:

```text
TypeError: scan body function carry input and carry output must have the same pytree structure
```

Root cause:

- Brax training wrappers carry `state.metrics` through `jax.lax.scan`
- the env reset/step metrics dict structures differed
- JAX requires identical pytree structure in scan carry

Fix:

```python
return State(..., metrics=state.metrics, ...)
```

Do not add custom changing keys to `state.metrics` inside the env step. Put
custom rollout metrics in `state.info`, external evaluation, or W&B callbacks.

## Current MJX Pipeline Direction

The project is moving away from scattered scripts toward a stable pipeline,
inspired by:

1. The user's previous notebook:

```text
C:\Users\12481\Desktop\pupper\RL\pupper-local-pipeline-main\Pupper_RL_CLOUD_now.ipynb
```

Useful patterns from that notebook:

- cloud setup: `MUJOCO_GL=auto`
- XLA Triton flag: `--xla_gpu_triton_gemm_any=True`
- JAX matmul precision: `jax_default_matmul_precision=high`
- PPO defaults from the Pupper notebook: `unroll_length=20`,
  `num_minibatches=32`, `num_updates_per_batch=4`, `batch_size=256`
- explicit policy net: `(256, 128, 128, 128)` with `elu`
- W&B config logging
- optional Brax `policy_params_fn(current_step, make_policy, params)` for
  visualization during training
- rollout video generated from `make_inference_fn(params)`

2. The user's classmate's working project:

```text
https://github.com/TundTT/bridge_mujoco_playground.git
```

This runs in the same `mjx312` environment and follows MuJoCo Playground style:
MJX + Brax/JAX PPO + registry/config/pipeline structure.

## In-Progress W&B Video Pipeline

The local pipeline changes add:

```text
robot_curl_mjx/pipeline.py
```

with helpers:

- `configure_cloud_runtime(...)`
- `hidden_layers_tuple(...)`
- `activation_fn(...)`
- `make_network_factory(...)`
- `render_policy_video(...)`

The training script is configured so that:

- default hidden layers are `256 128 128 128`
- default activation is `elu`
- XLA Triton, `jax_default_matmul_precision=high`, and `MUJOCO_GL=auto` are
  configured by the script
- `ppo.train(...)` receives an explicit `network_factory`
- `ppo.train(...)` receives `eval_env`
- the Brax policy callback is a callable no-op, so training never renders at
  intermediate evaluation points and avoids the earlier `NoneType` callback
  failure
- by default, training renders one `final_policy.mp4` after PPO finishes and
  uploads it to the still-open training W&B run when `--wandb` is enabled

The intended training command after this pipeline work is committed:

```bash
python -m scripts.mjx_train \
  --steps 200000 \
  --envs 512 \
  --episode-length 128 \
  --curl-goal 0.20 \
  --num-evals 5 \
  --wandb \
  --wandb-project robot-curl \
  --wandb-name mjx-curl-020
```

Use `--num-evals 1 --no-final-policy-video` only for diagnostic smoke tests. Brax
does an initial `steps=0` evaluation only when `num_evals > 1`, and that first
eval/JIT phase can dominate tiny runs. Normal training should keep enough
parallel envs/eval envs to use the GPU well. `--num-evals` controls metric
evaluation frequency only; it does not trigger video rendering.

## Playback / Evaluation

Pushed playback command:

```bash
python -m scripts.mjx_playback --params mjx_runs/curl_smoke/params --episodes 3 --no-video
```

OSMesa video command:

```bash
MUJOCO_GL=osmesa python -m scripts.mjx_playback \
  --params mjx_runs/curl_smoke/params \
  --episodes 3 \
  --video mjx_runs/curl_smoke/playback.mp4
```

Playback prints:

- `total_reward`
- `max_curl`
- `min_upright`
- `mean_contacts`
- `done`

and writes:

```text
mjx_runs/curl_smoke/eval.csv
mjx_runs/curl_smoke/playback.mp4
```

Important risk: playback must rebuild the same PPO network architecture used
during training. If hidden layers or activation change during training, pass the
same values to playback.

## Known Risks / Open Issues

1. The MJX env is still early.

The Brax env currently drives XML position actuators directly. It is not yet
feature-parity with the original Gym/MuJoCo env that uses explicit qfrc PD.

2. Reward may be exploitable.

The short run improves eval reward but does not prove the robot actually curls
in the desired tire-like posture. Always inspect video and rollout metrics.

3. W&B final policy video should be cloud-tested.

The local pipeline changes should be cloud-tested. Possible cloud failure
points:

- Brax final `make_inference_fn(params, deterministic=True)` signature
  compatibility
- Brax calling the no-op `policy_params_fn` with its expected three arguments
- `make_policy(params, deterministic=True)` signature compatibility
- OSMesa/imageio/ffmpeg availability
- rendering speed during training

4. `num_timesteps` may round up.

Brax PPO printed steps beyond the requested `--steps 10000` because rollout and
evaluation batch sizes round to training batch boundaries. This is normal.

5. The local machine is not the source of truth for MJX runtime behavior.

Local verification should focus on:

```bash
python -m py_compile ...
python -m scripts.mjx_train --help
python -m scripts.mjx_playback --help
```

Cloud verification should run actual MJX training/playback.

## Suggested Next Steps

1. Commit and push the local pipeline/final-video changes once verified.

2. On cloud, run a small W&B final-video test:

```bash
python -m scripts.mjx_train \
  --steps 10000 \
  --envs 128 \
  --episode-length 128 \
  --curl-goal 0.20 \
  --num-evals 1 \
  --no-final-policy-video \
  --wandb \
  --wandb-project robot-curl \
  --wandb-name mjx-speed-smoke
```

3. Then run a small final-video test:

```bash
python -m scripts.mjx_train ... --wandb --num-evals 2
```

This separates training correctness from rendering correctness.

4. Once video works, inspect whether the robot is truly curling upright.

Look for:

- not side-lying
- torso curl increasing
- feet repositioning instead of just dragging
- contact count not collapsing to zero
- no obvious reward hacking

5. Start reward/curriculum cleanup.

Likely next curriculum:

- Stage 0: stable standing under MJX env
- Stage 1: small torso curl target around `0.15-0.20 rad`
- Stage 2: increase curl goal
- Stage 3: introduce forward rolling objective

6. Longer-term architecture cleanup.

Move from ad hoc scripts toward:

```text
robot_curl_mjx/config.py
robot_curl_mjx/envs/
robot_curl_mjx/pipeline.py
robot_curl_mjx/render.py
scripts/mjx_train.py
scripts/mjx_eval.py
```

Keep the user's previous Pupper cloud notebook and `bridge_mujoco_playground`
as style references.
