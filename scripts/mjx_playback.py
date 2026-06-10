"""Evaluate a saved MJX/Brax PPO policy and optionally render a video."""
import argparse
import csv
from pathlib import Path

import numpy as np

from robot_curl.config_args import add_task_config_args, task_config_from_args
from robot_curl_mjx.pipeline import (
    activation_fn,
    configure_cloud_runtime,
    hidden_layers_tuple,
)


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
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 128, 128, 128])
    parser.add_argument("--activation", default="elu", choices=["relu", "tanh", "elu", "swish", "silu"])
    parser.add_argument("--mujoco-gl", default="osmesa")
    add_task_config_args(parser)
    parser.set_defaults(action_repeat=1, max_episode_steps=128)
    return parser.parse_args(argv)


def _load_brax_deps():
    try:
        import jax
        from brax.io import model as model_io
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(
            "MJX playback requires Brax/JAX. Activate the mjx312 conda environment "
            "or install brax, jax, mujoco, and mujoco-mjx."
        ) from exc
    return jax, model_io, ppo_networks


def _make_policy(args, env, params):
    """Builds an inference function directly from saved PPO parameters."""
    _, _, ppo_networks = _load_brax_deps()
    network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        policy_hidden_layer_sizes=hidden_layers_tuple(args.hidden_layers),
        activation=activation_fn(args.activation),
    )
    make_inference_fn = ppo_networks.make_inference_fn(network)
    return make_inference_fn(params, deterministic=args.deterministic)


def _rollout_episode(env, policy, key, episode_length, render_every=1):
    """Runs the whole MJX rollout in one compiled scan."""
    jax, _, _ = _load_brax_deps()
    import jax.numpy as jp

    render_every = max(1, int(render_every))

    def do_rollout(reset_key):
        state = env.reset(reset_key)
        initial_qpos = state.pipeline_state.qpos

        def scan_step(carry, _):
            state, rng, already_done = carry
            rng, action_key = jax.random.split(rng)
            action, _ = policy(state.obs, action_key)
            state = env.step(state, action)
            active = ~already_done
            done = state.done.astype(bool)
            sample = (
                state.pipeline_state.qpos,
                jp.where(active, state.reward, 0.0),
                env._curl_amount(state.pipeline_state),
                env._upright(state.pipeline_state),
                env._foot_contacts(state.pipeline_state).sum(),
                done,
                active,
            )
            return (state, rng, already_done | done), sample

        (_, _, _), trajectory = jax.lax.scan(
            scan_step,
            (state, reset_key, jp.array(False)),
            None,
            length=episode_length,
        )
        return initial_qpos, trajectory

    initial_qpos, trajectory = jax.jit(do_rollout)(key)
    initial_qpos, trajectory = jax.device_get((initial_qpos, trajectory))
    qpos, rewards, curls, uprights, contacts, dones, active = map(np.asarray, trajectory)
    active = active.astype(bool)
    steps = int(active.sum())
    valid_steps = max(steps, 1)
    summary = {
        "steps": steps,
        "total_reward": float(rewards[:valid_steps].sum()),
        "max_curl": float(curls[:valid_steps].max()),
        "min_upright": float(uprights[:valid_steps].min()),
        "mean_contacts": float(contacts[:valid_steps].mean()),
        "done": bool(dones[:valid_steps].any()),
    }
    frames = [np.asarray(initial_qpos)]
    frames.extend(qpos[:steps:render_every])
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
    """Renders a precomputed trajectory with MuJoCo's headless renderer."""
    if not qpos_frames:
        return
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
            data.qpos[:] = np.asarray(qpos)
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
        summary, frames = _rollout_episode(
            env,
            policy,
            key,
            args.episode_length,
            render_every=args.render_every,
        )
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
