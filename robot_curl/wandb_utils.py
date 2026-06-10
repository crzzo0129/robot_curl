"""Weights & Biases helpers for training entrypoints."""

from dataclasses import asdict


def add_wandb_args(parser):
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="robot-curl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-save-model", action="store_true")


def _training_config(args, task_config, script_name):
    config = asdict(task_config)
    config["script"] = script_name
    for name in [
        "steps",
        "seed",
        "envs",
        "device",
        "n_steps",
        "episode_length",
        "num_evals",
        "num_eval_envs",
        "batch_size",
        "unroll_length",
        "num_minibatches",
        "num_updates_per_batch",
        "learning_rate",
        "entropy_cost",
        "discounting",
        "reward_scaling",
        "settle_steps",
        "hidden_layers",
        "activation",
        "xla_triton",
        "mujoco_gl",
        "train_policy_videos",
        "final_policy_video",
        "video_width",
        "video_height",
        "video_fps",
        "video_camera",
        "out",
        "log_dir",
        "mjx",
    ]:
        if hasattr(args, name):
            value = getattr(args, name)
            config[name] = str(value) if name in {"out", "log_dir"} else value
    return config


def init_wandb_run(args, task_config, script_name, wandb_module=None, sync_tensorboard=True):
    if not getattr(args, "wandb", False):
        return None
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError("Install wandb or run without --wandb.") from exc

    kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_name,
        "tags": args.wandb_tags,
        "config": _training_config(args, task_config, script_name),
        "sync_tensorboard": sync_tensorboard,
    }
    if args.wandb_mode is not None:
        kwargs["mode"] = args.wandb_mode
    return wandb_module.init(**kwargs)


def build_wandb_callback(args):
    if not getattr(args, "wandb", False):
        return None
    try:
        from wandb.integration.sb3 import WandbCallback
    except ImportError as exc:
        raise RuntimeError("Install wandb or run without --wandb.") from exc
    return WandbCallback(
        gradient_save_freq=0,
        model_save_path=str(args.out) if getattr(args, "wandb_save_model", False) else None,
        model_save_freq=0,
        verbose=2,
    )


def finish_wandb_run(run, wandb_module=None):
    if run is None:
        return
    if wandb_module is None:
        import wandb as wandb_module
    wandb_module.finish()


def merge_callbacks(*callbacks):
    return [callback for callback in callbacks if callback is not None]
