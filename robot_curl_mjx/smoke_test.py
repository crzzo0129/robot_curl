"""Smoke test for the MJX robot curl backend."""
import argparse

from robot_curl.config_args import add_task_config_args, task_config_from_args


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    add_task_config_args(parser)
    args = parser.parse_args(argv)

    from robot_curl_mjx.env import QuadrupedCurlMJXEnv

    try:
        env = QuadrupedCurlMJXEnv(config=task_config_from_args(args), seed=args.seed)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    obs = env.reset()
    reward = 0.0
    terminated = truncated = False
    zero_action = env.jp.zeros(env.action_size)
    for _ in range(args.steps):
        obs, reward, terminated, truncated = env.step(zero_action)
        if terminated or truncated:
            break

    print(f"jax_backend={env.jax.default_backend()}")
    print(f"obs_shape={tuple(obs.shape)} action_size={env.action_size}")
    print(f"steps={env.step_count} reward={reward:.3f} terminated={terminated} truncated={truncated}")


if __name__ == "__main__":
    main()
