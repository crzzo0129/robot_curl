"""Evaluate a saved MJX/Brax PPO policy and optionally render OSMesa video."""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

from robot_curl.config_args import add_task_config_args, task_config_from_args
from robot_curl_mjx.pipeline import configure_cloud_runtime, hidden_layers_tuple, make_network_factory


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
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 128, 128, 128])
    parser.add_argument("--activation", default="elu", choices=["relu", "tanh", "elu", "swish", "silu"])
    parser.add_argument("--mujoco-gl", default="auto")
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
    """直接从 params 构建 inference function，不跑 ppo.train。"""
    _, model_io, _ = _load_brax_deps()
    import jax
    from robot_curl_mjx.pipeline import hidden_layers_tuple, activation_fn
    from brax.training.agents.ppo import networks as ppo_networks

    net = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        policy_hidden_layer_sizes=hidden_layers_tuple(args.hidden_layers),
        activation=activation_fn(args.activation),
    )
    key = jax.random.PRNGKey(args.seed)
    make_inference_fn = ppo_networks.make_inference_fn(net)
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
    """GPU 加速渲染：MuJoCo Renderer + EGL。"""
    if not qpos_frames:
        print("WARNING: qpos_frames is empty, skipping video")
        return
    print(f"Render start: frames={len(qpos_frames)} size={width}x{height} fps={fps} mu=GL={os.environ.get('MUJOCO_GL','unset')}", flush=True)
    try:
        import imageio.v2 as imageio
        import mujoco
        print("imports OK", flush=True)
    except ImportError as exc:
        raise SystemExit("Video rendering requires mujoco and imageio[ffmpeg].") from exc

    from robot_curl_mjx.brax_env import _XML_PATH
    print(f"model path={_XML_PATH}", flush=True)

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    print("loading model...", flush=True)
    model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    data = mujoco.MjData(model)
    print("creating renderer...", flush=True)
    renderer = mujoco.Renderer(model, width=width, height=height)
    print(f"renderer created, rendering {len(qpos_frames)} frames...", flush=True)
    pixels = []
    try:
        for i, qpos in enumerate(qpos_frames):
            if i % 20 == 0:
                print(f"  frame {i}/{len(qpos_frames)}", flush=True)
            if hasattr(qpos, 'qpos'):
                q = np.asarray(qpos.qpos)
            else:
                q = np.asarray(qpos)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            if camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=camera)
            pixels.append(renderer.render())
        print(f"render done, {len(pixels)} frames, encoding mp4...", flush=True)
    finally:
        renderer.close()
    imageio.mimsave(video_path, pixels, fps=fps)
    print(f"video saved: {video_path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    configure_cloud_runtime(xla_triton=False, mujoco_gl=args.mujoco_gl)
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
