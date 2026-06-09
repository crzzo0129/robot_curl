"""Shared helpers for the MJX training/playback pipeline."""
import os
from pathlib import Path


def hidden_layers_tuple(values):
    return tuple(int(value) for value in values)


def configure_cloud_runtime(xla_triton=True, mujoco_gl="osmesa"):
    """Applies cloud-friendly defaults used by the MJX training scripts."""
    if xla_triton:
        flags = os.environ.get("XLA_FLAGS", "")
        flag = "--xla_gpu_triton_gemm_any=True"
        if flag not in flags:
            os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()
    if mujoco_gl:
        os.environ.setdefault("MUJOCO_GL", mujoco_gl)


def activation_fn(name):
    try:
        import jax.nn as jnn
    except ImportError as exc:
        raise RuntimeError("Activation functions require JAX.") from exc

    activations = {
        "relu": jnn.relu,
        "tanh": jnn.tanh,
        "elu": jnn.elu,
        "swish": jnn.swish,
        "silu": jnn.silu,
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unknown activation: {name}") from exc


def make_network_factory(hidden_layers, activation):
    try:
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise RuntimeError("Network factory requires Brax.") from exc

    return lambda *args, **kwargs: ppo_networks.make_ppo_networks(
        *args,
        policy_hidden_layer_sizes=hidden_layers_tuple(hidden_layers),
        activation=activation_fn(activation),
        **kwargs,
    )


def make_policy_video_callback(
    *,
    enabled,
    wandb_run,
    eval_env,
    out_dir,
    episode_length,
    seed,
    width,
    height,
    fps,
    camera,
):
    if not enabled or wandb_run is None:
        return None

    out_dir = Path(out_dir)

    def policy_params_fn(current_step, make_policy, params):
        import wandb
        from scripts.mjx_playback import _render_video, _rollout_episode

        try:
            import jax

            try:
                policy = make_policy(params, deterministic=True)
            except TypeError:
                policy = make_policy(params)
            key = jax.random.PRNGKey(seed + int(current_step))
            summary, frames = _rollout_episode(eval_env, policy, key, episode_length)
            video_path = out_dir / f"policy_step_{int(current_step):09d}.mp4"
            _render_video(video_path, frames, width, height, fps, camera)
            wandb.log(
                {
                    "policy_video": wandb.Video(str(video_path), fps=fps, format="mp4"),
                    "policy_video/max_curl": summary["max_curl"],
                    "policy_video/min_upright": summary["min_upright"],
                    "policy_video/total_reward": summary["total_reward"],
                    "policy_video/mean_contacts": summary["mean_contacts"],
                },
                step=int(current_step),
            )
            print(f"stage=wandb_video step={current_step} path={video_path}", flush=True)
        except Exception as exc:
            print(f"stage=wandb_video_failed step={current_step} error={exc}", flush=True)

    return policy_params_fn
