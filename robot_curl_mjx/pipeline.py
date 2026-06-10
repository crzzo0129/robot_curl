"""Shared helpers for the MJX training/playback pipeline."""
import os
from pathlib import Path


def hidden_layers_tuple(values):
    return tuple(int(value) for value in values)


def _detect_mujoco_gl_backend():
    try:
        import mujoco
    except ImportError:
        return None

    test_xml = '<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    old_backend = os.environ.get("MUJOCO_GL")
    for backend in ("egl", "osmesa"):
        renderer = None
        try:
            os.environ["MUJOCO_GL"] = backend
            model = mujoco.MjModel.from_xml_string(test_xml)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, 32, 32)
            renderer.update_scene(data)
            pixels = renderer.render()
            if pixels is not None and pixels.size > 0:
                return backend
        except Exception:
            continue
        finally:
            if renderer is not None:
                renderer.close()
            if old_backend is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = old_backend
    return None


def configure_cloud_runtime(xla_triton=True, mujoco_gl="auto", matmul_precision="high", verbose=False):
    """Applies cloud-friendly defaults used by the MJX training scripts."""
    if xla_triton:
        flags = os.environ.get("XLA_FLAGS", "")
        flag = "--xla_gpu_triton_gemm_any=True"
        if flag not in flags:
            os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()

    if mujoco_gl == "auto":
        backend = _detect_mujoco_gl_backend()
        os.environ["MUJOCO_GL"] = backend or "disabled"
    elif mujoco_gl:
        os.environ.setdefault("MUJOCO_GL", mujoco_gl)

    if matmul_precision:
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", matmul_precision)
        try:
            import jax

            jax.config.update("jax_default_matmul_precision", matmul_precision)
        except ImportError:
            pass

    if verbose:
        backend = "unavailable"
        try:
            import jax

            backend = jax.default_backend()
        except Exception:
            pass
        print(
            "stage=runtime_config "
            f"jax_backend={backend} "
            f"mujoco_gl={os.environ.get('MUJOCO_GL', '')} "
            f"matmul_precision={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', '')} "
            f"xla_flags={os.environ.get('XLA_FLAGS', '')}",
            flush=True,
        )


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


def render_policy_video(
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
    step,
    make_policy,
    params,
    video_name=None,
    metric_prefix="policy_video",
):
    if not enabled:
        return None

    out_dir = Path(out_dir)
    from scripts.mjx_playback import _render_video, _rollout_episode

    import jax

    try:
        policy = make_policy(params, deterministic=True)
    except TypeError:
        policy = make_policy(params)
    key = jax.random.PRNGKey(seed + int(step))
    summary, frames = _rollout_episode(eval_env, policy, key, episode_length)
    if video_name is None:
        video_name = f"policy_step_{int(step):09d}.mp4"
    video_path = out_dir / video_name
    _render_video(video_path, frames, width, height, fps, camera)

    if wandb_run is not None:
        import wandb

        wandb.log(
            {
                metric_prefix: wandb.Video(str(video_path), fps=fps, format="mp4"),
                f"{metric_prefix}/step": int(step),
                f"{metric_prefix}/max_curl": summary["max_curl"],
                f"{metric_prefix}/min_upright": summary["min_upright"],
                f"{metric_prefix}/total_reward": summary["total_reward"],
                f"{metric_prefix}/mean_contacts": summary["mean_contacts"],
            }
        )
    print(f"stage={metric_prefix} step={step} path={video_path}", flush=True)
    return summary, video_path


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
        def policy_params_noop(current_step, make_policy, params):
            return None

        return policy_params_noop

    def policy_params_fn(current_step, make_policy, params):
        try:
            render_policy_video(
                enabled=True,
                wandb_run=wandb_run,
                eval_env=eval_env,
                out_dir=out_dir,
                episode_length=episode_length,
                seed=seed,
                width=width,
                height=height,
                fps=fps,
                camera=camera,
                step=current_step,
                make_policy=make_policy,
                params=params,
            )
        except Exception as exc:
            print(f"stage=policy_video_failed step={current_step} error={exc}", flush=True)

    return policy_params_fn
