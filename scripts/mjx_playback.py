"""Evaluate a saved MJX/Brax PPO policy and optionally render OSMesa video."""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

from robot_curl.config_args import add_task_config_args, task_config_from_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="mjx_runs/curl_smoke/params")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--csv", default="mjx_runs/curl_smoke/eval.csv")
    parser.add_argument("--video", default="mjx_runs/curl_smoke/playback.mp4")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-episode", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", default=None)
    add_task_config_args(parser)
    parser.set_defaults(action_repeat=1, max_episode_steps=128)
    return parser.parse_args(argv)


def _load_brax_deps():
    try:
        import jax
        from brax.io import model as model_io
        from brax.training.agents.ppo import train as ppo
    except ImportError as exc:
        raise SystemExit(
            "MJX playback requires Brax/JAX. Activate the mjx312 conda environment "
            "or install brax, jax, mujoco, and mujoco-mjx."
        ) from exc
    return jax, model_io, ppo


def _make_policy(args, env, params):
    _, _, ppo = _load_brax_deps()
    make_inference_fn, _, _ = ppo.train(
        environment=env,
        num_timesteps=0,
        episode_length=args.episode_length,
        action_repeat=1,
        num_envs=1,
        num_evals=0,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        reward_scaling=1.0,
        unroll_length=1,
        batch_size=1,
        num_minibatches=1,
        num_updates_per_batch=1,
        normalize_observations=True,
        seed=args.seed,
        progress_fn=lambda *_: None,
    )
    return make_inference_fn(params, deterministic=args.deterministic)


def _to_float(value):
    return float(np.asarray(value))


def _rollout_episode(env, policy, key, episode_length):
    jax, _, _ = _load_brax_deps()
    state = env.reset(key)
    frames = [np.asarray(state.pipeline_state.qpos)]
    total_reward = 0.0
    max_curl = _to_float(env._curl_amount(state.pipeline_state))
    min_upright = _to_float(env._upright(state.pipeline_state))
    mean_contacts = 0.0
    steps = 0
    done = False

    for step in range(episode_length):
        key, action_key = jax.random.split(key)
        action, _ = policy(state.obs, action_key)
        state = env.step(state, action)
        state.reward.block_until_ready()
        steps = step + 1
        total_reward += _to_float(state.reward)
        curl = _to_float(env._curl_amount(state.pipeline_state))
        upright = _to_float(env._upright(state.pipeline_state))
        contacts = _to_float(env._foot_contacts(state.pipeline_state).sum())
        max_curl = max(max_curl, curl)
        min_upright = min(min_upright, upright)
        mean_contacts += contacts
        frames.append(np.asarray(state.pipeline_state.qpos))
        done = bool(np.asarray(state.done))
        if done:
            break

    mean_contacts = mean_contacts / max(steps, 1)
    summary = {
        "steps": steps,
        "total_reward": total_reward,
        "max_curl": max_curl,
        "min_upright": min_upright,
        "mean_contacts": mean_contacts,
        "done": done,
    }
    return summary, frames


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["episode", "steps", "total_reward", "max_curl", "min_upright", "mean_contacts", "done"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _render_video(video_path, qpos_frames, width, height, fps, camera):
    if not qpos_frames:
        return
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    try:
        import imageio.v2 as imageio
        import mujoco
    except ImportError as exc:
        raise SystemExit("Video rendering requires mujoco and imageio[ffmpeg].") from exc

    from robot_curl_mjx.brax_env import _XML_PATH

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    pixels = []
    try:
        for qpos in qpos_frames:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            if camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=camera)
            pixels.append(renderer.render())
    finally:
        renderer.close()
    imageio.mimsave(video_path, pixels, fps=fps)


def main(argv=None):
    args = parse_args(argv)
    task_config = task_config_from_args(args)
    jax, model_io, _ = _load_brax_deps()

    from robot_curl_mjx.brax_env import make_brax_env

    env = make_brax_env(config=task_config, seed=args.seed, settle_steps=args.settle_steps)
    params = model_io.load_params(args.params)
    policy = _make_policy(args, env, params)

    rows = []
    video_frames = None
    for episode in range(args.episodes):
        key = jax.random.PRNGKey(args.seed + episode)
        summary, frames = _rollout_episode(env, policy, key, args.episode_length)
        row = {"episode": episode, **summary}
        rows.append(row)
        print(
            "episode={episode} steps={steps} total_reward={total_reward:.3f} "
            "max_curl={max_curl:.3f} min_upright={min_upright:.3f} "
            "mean_contacts={mean_contacts:.3f} done={done}".format(**row),
            flush=True,
        )
        if episode == args.video_episode:
            video_frames = frames

    _write_csv(args.csv, rows)
    print(f"saved_csv={args.csv}", flush=True)
    if not args.no_video:
        _render_video(args.video, video_frames, args.width, args.height, args.fps, args.camera)
        print(f"saved_video={args.video}", flush=True)


if __name__ == "__main__":
    main()
