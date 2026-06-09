"""Smoke test for the MJX robot curl backend."""
import argparse
from dataclasses import replace

from robot_curl.config_args import add_task_config_args, task_config_from_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--skip-reward", action="store_true")
    parser.add_argument("--skip-terminated", action="store_true")
    add_task_config_args(parser)
    parser.set_defaults(action_repeat=1, max_episode_steps=10)
    return parser.parse_args(argv)


def _block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _diagnostic_step(env, action, args):
    print("stage=action_prepare", flush=True)
    action = env.jp.clip(env.jp.asarray(action), -env.config.action_scale, env.config.action_scale)
    env.target_q = env.jp.clip(env.target_q + action, env.jnt_low_jp, env.jnt_high_jp)
    _block_until_ready(env.target_q)
    print("stage=action_ready", flush=True)

    for repeat in range(env.config.action_repeat):
        print(f"stage=apply_pd_start repeat={repeat + 1}", flush=True)
        env._apply_pd()
        _block_until_ready(env.data.qfrc_applied)
        print(f"stage=apply_pd_done repeat={repeat + 1}", flush=True)

        print(f"stage=mjx_step_start repeat={repeat + 1}", flush=True)
        env.data = env.mjx.step(env.mjx_model, env.data)
        print(f"stage=mjx_step_returned repeat={repeat + 1}", flush=True)
        _block_until_ready(env.data.qpos)
        print(f"stage=mjx_step_ready repeat={repeat + 1}", flush=True)

    env.step_count += 1
    reward = 0.0
    terminated = False

    if not args.skip_reward:
        print("stage=reward_start", flush=True)
        reward = env.reward(action)
        print(f"stage=reward_done reward={reward:.3f}", flush=True)

    if not args.skip_terminated:
        print("stage=terminated_start", flush=True)
        terminated = env.terminated()
        print(f"stage=terminated_done terminated={terminated}", flush=True)

    truncated = env.step_count >= env.config.max_episode_steps
    print("stage=obs_start", flush=True)
    obs = env.obs()
    _block_until_ready(obs)
    print(f"stage=obs_done obs_shape={tuple(obs.shape)}", flush=True)
    return obs, reward, terminated, truncated


def main(argv=None):
    args = parse_args(argv)

    from robot_curl_mjx.env import QuadrupedCurlMJXEnv

    task_config = task_config_from_args(args)
    task_config = replace(task_config, action_repeat=args.action_repeat, max_episode_steps=args.max_episode_steps)
    print("stage=init_env", flush=True)
    try:
        env = QuadrupedCurlMJXEnv(config=task_config, seed=args.seed, settle_steps=args.settle_steps)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"stage=env_ready jax_backend={env.jax.default_backend()}", flush=True)
    print("stage=reset_start", flush=True)
    obs = env.reset()
    print(f"stage=reset_done obs_shape={tuple(obs.shape)}", flush=True)
    reward = 0.0
    terminated = truncated = False
    zero_action = env.jp.zeros(env.action_size)
    for step in range(args.steps):
        print(f"stage=step_start step={step + 1}", flush=True)
        obs, reward, terminated, truncated = _diagnostic_step(env, zero_action, args)
        print(f"stage=step_done step={env.step_count} reward={reward:.3f} terminated={terminated} truncated={truncated}", flush=True)
        if terminated or truncated:
            break

    print(f"jax_backend={env.jax.default_backend()}")
    print(f"obs_shape={tuple(obs.shape)} action_size={env.action_size}")
    print(f"steps={env.step_count} reward={reward:.3f} terminated={terminated} truncated={truncated}")


if __name__ == "__main__":
    main()
