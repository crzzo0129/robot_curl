"""MJX/Brax PPO training entrypoint for robot curl."""
import argparse
from pathlib import Path

from robot_curl.config_args import add_task_config_args, task_config_from_args
from robot_curl.wandb_utils import add_wandb_args, finish_wandb_run, init_wandb_run


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--num-evals", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-length", type=int, default=10)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-2)
    parser.add_argument("--discounting", type=float, default=0.97)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("mjx_runs") / "curl_smoke")
    add_task_config_args(parser)
    add_wandb_args(parser)
    parser.set_defaults(action_repeat=1, max_episode_steps=128)
    return parser.parse_args(argv)


def _metric_to_float(value):
    try:
        return float(value)
    except TypeError:
        return float(value.item())


def _make_progress_fn(wandb_run):
    def progress(num_steps, metrics):
        clean_metrics = {name: _metric_to_float(value) for name, value in metrics.items()}
        reward = clean_metrics.get("eval/episode_reward", clean_metrics.get("eval/episode_reward_mean"))
        message = f"steps={num_steps}"
        if reward is not None:
            message += f" eval_reward={reward:.3f}"
        if "eval/episode_length" in clean_metrics:
            message += f" eval_length={clean_metrics['eval/episode_length']:.1f}"
        print(message, flush=True)

        if wandb_run is not None:
            import wandb

            wandb.log(clean_metrics, step=int(num_steps))

    return progress


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    task_config = task_config_from_args(args)

    try:
        from brax.io import model as model_io
        from brax.training.agents.ppo import train as ppo
    except ImportError as exc:
        raise SystemExit(
            "MJX training requires Brax. Activate the mjx312 conda environment "
            "or install brax, jax, mujoco, and mujoco-mjx."
        ) from exc
    from robot_curl_mjx.brax_env import make_brax_env

    env = make_brax_env(config=task_config, seed=args.seed, settle_steps=args.settle_steps)
    wandb_run = init_wandb_run(args, task_config, script_name="mjx_train")
    try:
        print(
            "stage=train_start "
            f"steps={args.steps} envs={args.envs} episode_length={args.episode_length} "
            f"action_repeat={task_config.action_repeat}",
            flush=True,
        )
        train_result = ppo.train(
            environment=env,
            num_timesteps=args.steps,
            episode_length=args.episode_length,
            action_repeat=1,
            num_envs=args.envs,
            num_evals=args.num_evals,
            learning_rate=args.learning_rate,
            entropy_cost=args.entropy_cost,
            discounting=args.discounting,
            reward_scaling=args.reward_scaling,
            unroll_length=args.unroll_length,
            batch_size=args.batch_size,
            num_minibatches=args.num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            normalize_observations=True,
            seed=args.seed,
            progress_fn=_make_progress_fn(wandb_run),
        )
        make_inference_fn, params, metrics = train_result
        model_io.save_params(args.out / "params", params)
        print(f"stage=train_done saved={args.out / 'params'}", flush=True)
        if metrics:
            final_metrics = {name: _metric_to_float(value) for name, value in metrics.items()}
            print(f"final_metrics={final_metrics}", flush=True)
        return make_inference_fn, params
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
