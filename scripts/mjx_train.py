"""MJX/Brax PPO training entrypoint for robot curl."""
import argparse
import json
import time
from pathlib import Path

from robot_curl.config_args import add_task_config_args, task_config_from_args
from robot_curl.wandb_utils import add_wandb_args, finish_wandb_run, init_wandb_run
from robot_curl_mjx.pipeline import (
    configure_cloud_runtime,
    hidden_layers_tuple,
    make_network_factory,
    render_policy_video,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-2)
    parser.add_argument("--discounting", type=float, default=0.97)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("mjx_runs") / "curl_smoke")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 128, 128, 128])
    parser.add_argument("--activation", default="elu", choices=["relu", "tanh", "elu", "swish", "silu"])
    parser.add_argument("--xla-triton", action="store_true", default=True)
    parser.add_argument("--no-xla-triton", dest="xla_triton", action="store_false")
    parser.add_argument("--mujoco-gl", default="osmesa")
    parser.add_argument(
        "--matmul-precision",
        default="high",
        choices=["default", "high", "highest", "bfloat16", "tensorfloat32"],
    )
    parser.add_argument("--runtime-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-runtime-diagnostics", dest="runtime_diagnostics", action="store_false")
    parser.add_argument("--final-policy-video", action="store_true", default=True)
    parser.add_argument("--no-final-policy-video", dest="final_policy_video", action="store_false")
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-render-every", type=int, default=2)
    parser.add_argument("--video-camera", default=None)
    add_task_config_args(parser)
    add_wandb_args(parser)
    parser.set_defaults(action_repeat=1, max_episode_steps=128)
    return parser.parse_args(argv)


def _metric_to_float(value):
    try:
        return float(value)
    except TypeError:
        return float(value.item())


def _make_policy_params_fn(best_state):
    def policy_params(current_step, make_policy, params):
        del make_policy
        step = int(current_step)
        best_state["candidate_step"] = step
        best_state["candidate_params"] = params
        if best_state["step"] == step:
            best_state["params"] = params

    return policy_params


def _configure_wandb_metrics(wandb_run):
    if wandb_run is None:
        return
    wandb_run.define_metric("train_step")
    wandb_run.define_metric("training/*", step_metric="train_step")
    wandb_run.define_metric("eval/*", step_metric="train_step")


def _make_progress_fn(wandb_run, progress_times, metrics_history=None, best_state=None):
    def progress(num_steps, metrics):
        progress_times.append((int(num_steps), time.perf_counter()))
        clean_metrics = {name: _metric_to_float(value) for name, value in metrics.items()}
        if metrics_history is not None:
            metrics_history.append({"step": int(num_steps), **clean_metrics})
        reward = clean_metrics.get("eval/episode_reward", clean_metrics.get("eval/episode_reward_mean"))
        if best_state is not None and reward is not None and reward > best_state["reward"]:
            best_state["reward"] = reward
            best_state["step"] = int(num_steps)
            if best_state["candidate_step"] == int(num_steps):
                best_state["params"] = best_state["candidate_params"]
        message = f"steps={num_steps}"
        if reward is not None:
            message += f" eval_reward={reward:.3f}"
        if "eval/episode_length" in clean_metrics:
            message += f" eval_length={clean_metrics['eval/episode_length']:.1f}"
        print(message, flush=True)

        if wandb_run is not None:
            clean_metrics["train_step"] = int(num_steps)
            wandb_run.log(clean_metrics)

    return progress


def _write_metrics_history(path, metrics_history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_history, fh, indent=2)


def _log_final_metrics(wandb_run, metrics, train_step):
    if wandb_run is None or not metrics:
        return
    wandb_run.log({**metrics, "train_step": int(train_step)})


def _print_timing_summary(train_start, train_end, progress_times, wandb_run):
    if not progress_times:
        total = train_end - train_start
        print(f"time_total={total:.3f}s", flush=True)
        if wandb_run is not None:
            wandb_run.summary["time_total"] = total
        return

    first_step, first_progress_time = progress_times[0]
    time_to_first_progress = first_progress_time - train_start
    time_after_first_progress = train_end - first_progress_time
    print(
        "stage=timing "
        f"first_progress_step={first_step} "
        f"time_to_first_progress={time_to_first_progress:.3f}s "
        f"time_after_first_progress={time_after_first_progress:.3f}s "
        f"time_total={train_end - train_start:.3f}s",
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.summary["time_to_first_progress"] = time_to_first_progress
        wandb_run.summary["time_after_first_progress"] = time_after_first_progress
        wandb_run.summary["time_total"] = train_end - train_start


def main(argv=None):
    args = parse_args(argv)
    configure_cloud_runtime(
        xla_triton=args.xla_triton,
        mujoco_gl=args.mujoco_gl,
        matmul_precision=args.matmul_precision,
        verbose=args.runtime_diagnostics,
    )
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
    eval_env = make_brax_env(config=task_config, seed=args.seed + 10_000, settle_steps=args.settle_steps)
    wandb_run = init_wandb_run(args, task_config, script_name="mjx_train", sync_tensorboard=False)
    _configure_wandb_metrics(wandb_run)
    try:
        print(
            "stage=train_start "
            f"steps={args.steps} envs={args.envs} episode_length={args.episode_length} "
            f"action_repeat={task_config.action_repeat}",
            flush=True,
        )
        progress_times = []
        metrics_history = []
        best_state = {
            "reward": float("-inf"),
            "step": None,
            "params": None,
            "candidate_step": None,
            "candidate_params": None,
        }
        train_start_time = time.perf_counter()
        train_result = ppo.train(
            environment=env,
            eval_env=eval_env,
            num_timesteps=args.steps,
            episode_length=args.episode_length,
            action_repeat=1,
            num_envs=args.envs,
            num_evals=args.num_evals,
            num_eval_envs=args.num_eval_envs,
            learning_rate=args.learning_rate,
            entropy_cost=args.entropy_cost,
            discounting=args.discounting,
            reward_scaling=args.reward_scaling,
            unroll_length=args.unroll_length,
            batch_size=args.batch_size,
            num_minibatches=args.num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            normalize_observations=True,
            network_factory=make_network_factory(hidden_layers_tuple(args.hidden_layers), args.activation),
            seed=args.seed,
            progress_fn=_make_progress_fn(wandb_run, progress_times, metrics_history, best_state),
            policy_params_fn=_make_policy_params_fn(best_state),
        )
        train_end_time = time.perf_counter()
        _print_timing_summary(train_start_time, train_end_time, progress_times, wandb_run)
        make_inference_fn, params, metrics = train_result
        best_params = best_state["params"] if best_state["params"] is not None else params
        best_step = best_state["step"] if best_state["step"] is not None else args.steps
        model_io.save_params(args.out / "params_final", params)
        model_io.save_params(args.out / "params_best", best_params)
        model_io.save_params(args.out / "params", best_params)
        metrics_path = args.out / "metrics_history.json"
        _write_metrics_history(metrics_path, metrics_history)
        print(f"stage=metrics_saved path={metrics_path}", flush=True)
        print(
            "stage=train_done "
            f"saved_best={args.out / 'params_best'} "
            f"saved_final={args.out / 'params_final'} "
            f"best_step={best_step} best_reward={best_state['reward']:.3f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.summary["best_eval_reward"] = best_state["reward"]
            wandb_run.summary["best_eval_step"] = best_step
        if metrics:
            final_metrics = {name: _metric_to_float(value) for name, value in metrics.items()}
            print(f"final_metrics={final_metrics}", flush=True)
            final_step = progress_times[-1][0] if progress_times else args.steps
            _log_final_metrics(wandb_run, final_metrics, final_step)
            print(f"stage=wandb_metrics_logged step={final_step}", flush=True)
        try:
            render_policy_video(
                enabled=args.final_policy_video,
                wandb_run=wandb_run,
                eval_env=eval_env,
                out_dir=args.out / "videos",
                episode_length=args.episode_length,
                seed=args.seed,
                width=args.video_width,
                height=args.video_height,
                fps=args.video_fps,
                camera=args.video_camera,
                step=best_step,
                make_policy=make_inference_fn,
                params=best_params,
                render_every=args.video_render_every,
                video_name="final_policy.mp4",
                metric_prefix="final_policy_video",
            )
        except Exception as exc:
            print(f"stage=final_policy_video_failed error={exc}", flush=True)
        return make_inference_fn, best_params
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
