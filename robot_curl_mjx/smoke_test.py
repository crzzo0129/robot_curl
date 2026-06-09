"""Smoke test for the MJX robot curl backend."""
import argparse
from dataclasses import replace

from robot_curl.config_args import add_task_config_args, task_config_from_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    add_task_config_args(parser)
    parser.set_defaults(action_repeat=1, max_episode_steps=10)
    return parser.parse_args(argv)


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
        obs, reward, terminated, truncated = env.step(zero_action)
        print(f"stage=step_done step={env.step_count} reward={reward:.3f} terminated={terminated} truncated={truncated}", flush=True)
        if terminated or truncated:
            break

    print(f"jax_backend={env.jax.default_backend()}")
    print(f"obs_shape={tuple(obs.shape)} action_size={env.action_size}")
    print(f"steps={env.step_count} reward={reward:.3f} terminated={terminated} truncated={truncated}")


if __name__ == "__main__":
    main()
