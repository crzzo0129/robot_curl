def test_mjx_train_entry_imports_without_initializing_gpu():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys; import scripts.mjx_train; print('mujoco' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"

    from scripts import mjx_train

    assert callable(mjx_train.main)


def test_mjx_playback_entry_imports_without_initializing_gpu():
    from scripts import mjx_playback

    assert callable(mjx_playback.main)


def test_mjx_train_defaults_are_gpu_smoke_sized():
    from scripts.mjx_train import parse_args

    args = parse_args([])

    assert args.steps == 10_000
    assert args.envs == 128
    assert args.episode_length == 128
    assert args.num_evals == 10
    assert args.num_eval_envs == 128
    assert args.unroll_length == 20
    assert args.num_minibatches == 32
    assert args.action_repeat == 1
    assert args.hidden_layers == [256, 128, 128, 128]
    assert args.activation == "elu"
    assert args.mujoco_gl == "osmesa"
    assert args.matmul_precision == "high"
    assert args.runtime_diagnostics is True
    assert not hasattr(args, "train_policy_videos")
    assert args.final_policy_video is True
    assert args.video_width == 320
    assert args.video_height == 240
    assert args.reward_leg_fold == 0.5


def test_mjx_train_accepts_leg_fold_reward_weight():
    from scripts.mjx_train import parse_args

    assert parse_args(["--reward-leg-fold", "0.8"]).reward_leg_fold == 0.8


def test_mjx_progress_logs_with_environment_step():
    from scripts.mjx_train import _make_progress_fn

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, metrics, **kwargs):
            self.logged.append((metrics, kwargs))

    run = FakeRun()
    progress_times = []
    progress = _make_progress_fn(run, progress_times)

    progress(51200, {"eval/episode_reward": 12.5})

    assert progress_times[0][0] == 51200
    assert run.logged[0][1] == {}
    assert run.logged[0][0]["train_step"] == 51200
    assert run.logged[0][0]["eval/episode_reward"] == 12.5


def test_mjx_wandb_metrics_use_train_step_axis():
    from scripts.mjx_train import _configure_wandb_metrics

    class FakeRun:
        def __init__(self):
            self.defined = []

        def define_metric(self, name, **kwargs):
            self.defined.append((name, kwargs))

    run = FakeRun()

    _configure_wandb_metrics(run)

    assert ("train_step", {}) in run.defined
    assert ("training/*", {"step_metric": "train_step"}) in run.defined
    assert ("eval/*", {"step_metric": "train_step"}) in run.defined


def test_mjx_final_metrics_are_committed_before_video():
    from scripts.mjx_train import _log_final_metrics

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, metrics, **kwargs):
            self.logged.append((metrics, kwargs))

    run = FakeRun()

    _log_final_metrics(run, {"training/sps": 42.0}, train_step=655360)

    assert run.logged == [({"training/sps": 42.0, "train_step": 655360}, {})]


def test_mjx_pipeline_parses_hidden_layers():
    from robot_curl_mjx.pipeline import hidden_layers_tuple

    assert hidden_layers_tuple([64, 32]) == (64, 32)


def test_mjx_pipeline_configures_cloud_runtime(monkeypatch):
    from robot_curl_mjx.pipeline import configure_cloud_runtime

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "")
    configure_cloud_runtime(xla_triton=True, mujoco_gl="osmesa", matmul_precision=None)

    assert "xla_gpu_triton_gemm_any=True" in __import__("os").environ["XLA_FLAGS"]
    assert __import__("os").environ["MUJOCO_GL"] == "osmesa"


def test_mjx_pipeline_auto_selects_osmesa_for_headless_linux(monkeypatch):
    from robot_curl_mjx.pipeline import select_mujoco_gl_backend

    environ = {}

    assert select_mujoco_gl_backend(environ=environ, platform_name="linux") == "osmesa"


def test_mjx_pipeline_auto_preserves_explicit_backend():
    from robot_curl_mjx.pipeline import select_mujoco_gl_backend

    assert select_mujoco_gl_backend(environ={"MUJOCO_GL": "osmesa"}, platform_name="linux") == "osmesa"


def test_mjx_pipeline_sets_pyopengl_platform(monkeypatch):
    from robot_curl_mjx.pipeline import configure_cloud_runtime

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    configure_cloud_runtime(xla_triton=False, mujoco_gl="osmesa", matmul_precision=None)

    assert __import__("os").environ["MUJOCO_GL"] == "osmesa"
    assert __import__("os").environ["PYOPENGL_PLATFORM"] == "osmesa"


def test_mjx_pipeline_configures_matmul_precision(monkeypatch):
    from robot_curl_mjx.pipeline import configure_cloud_runtime

    monkeypatch.delenv("JAX_DEFAULT_MATMUL_PRECISION", raising=False)
    configure_cloud_runtime(xla_triton=False, mujoco_gl=None, matmul_precision="high")

    assert __import__("os").environ["JAX_DEFAULT_MATMUL_PRECISION"] == "high"


def test_mjx_playback_defaults_use_osmesa_video_path():
    from scripts.mjx_playback import parse_args

    args = parse_args([])

    assert args.episodes == 3
    assert args.episode_length == 128
    assert args.params == "mjx_runs/curl_smoke/params"
    assert args.video == "mjx_runs/curl_smoke/playback.mp4"
    assert args.mujoco_gl == "osmesa"
    assert args.width == 320
    assert args.height == 240
    assert args.render_every == 2
    assert not hasattr(args, "wandb")


def test_mjx_playback_rollout_uses_jitted_scan():
    import inspect

    from scripts.mjx_playback import _rollout_episode

    source = inspect.getsource(_rollout_episode)

    assert "jax.lax.scan" in source
    assert "jax.jit" in source
    assert "for step in range" not in source


def test_mjx_progress_tracks_best_eval_reward():
    from scripts.mjx_train import _make_progress_fn

    best = {
        "reward": float("-inf"),
        "step": None,
        "params": None,
        "candidate_step": 100,
        "candidate_params": "params-100",
    }
    progress = _make_progress_fn(None, [], best_state=best)

    progress(100, {"eval/episode_reward": 3.0})
    assert best["reward"] == 3.0
    assert best["step"] == 100
    assert best["params"] == "params-100"

    progress(200, {"eval/episode_reward": 2.0})
    assert best["reward"] == 3.0
    assert best["step"] == 100


def test_mjx_policy_callback_pairs_initial_best_with_params():
    from scripts.mjx_train import _make_policy_params_fn

    best = {
        "reward": 4.5,
        "step": 0,
        "params": None,
        "candidate_step": None,
        "candidate_params": None,
    }
    callback = _make_policy_params_fn(best)

    params = object()
    callback(0, object(), params)

    assert best["candidate_step"] == 0
    assert best["candidate_params"] is params
    assert best["params"] is params


def test_mjx_brax_env_declares_fixed_reward_metrics():
    source = open("robot_curl_mjx/brax_env.py", encoding="utf-8").read()

    for metric in [
        "reward",
        "reward_total",
        "reward_curl",
        "reward_progress",
        "reward_tier",
        "reward_contact",
        "reward_leg_fold",
        "reward_smooth",
        "reward_stable",
        "reward_upright",
        "reward_overcurl",
        "reward_alive",
    ]:
        assert f'"{metric}"' in source

    assert "metrics=reward_metrics" in source
    assert '"reward": reward' in source


def test_final_policy_video_logs_once_to_supplied_training_run(tmp_path):
    from robot_curl_mjx.pipeline import _log_policy_video

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, payload):
            self.logged.append(payload)

    class FakeWandb:
        @staticmethod
        def Video(path, fps, format):
            return {"path": path, "fps": fps, "format": format}

    run = FakeRun()
    video_path = tmp_path / "final_policy.mp4"
    summary = {
        "max_curl": 0.25,
        "min_upright": 0.75,
        "total_reward": 12.0,
        "mean_contacts": 3.0,
    }

    _log_policy_video(
        wandb_run=run,
        video_path=video_path,
        fps=30,
        metric_prefix="final_policy_video",
        step=10_000,
        summary=summary,
        wandb_module=FakeWandb,
    )

    assert len(run.logged) == 1
    payload = run.logged[0]
    assert payload["final_policy_video"]["path"] == str(video_path)
    assert payload["final_policy_video/step"] == 10_000
    assert payload["final_policy_video/total_reward"] == 12.0


def test_mjx_brax_env_factory_imports_without_heavy_dependencies():
    from robot_curl_mjx.brax_env import make_brax_env

    assert callable(make_brax_env)


def test_robot_model_removes_hip_abduction_dofs():
    import xml.etree.ElementTree as ET

    from robot_curl.task_config import CurlTaskConfig, JOINT_NAMES, N_JOINTS

    root = ET.parse("assets/quadruped.xml").getroot()
    joint_names = {node.get("name") for node in root.findall(".//joint")}
    actuator_joints = {node.get("joint") for node in root.findall("./actuator/*")}
    removed = {"fl_hip_abd", "fr_hip_abd", "hl_hip_abd", "hr_hip_abd"}

    assert joint_names.isdisjoint(removed)
    assert actuator_joints.isdisjoint(removed)
    assert all("hip_abd" not in name for name in JOINT_NAMES)
    assert N_JOINTS == 9
    assert CurlTaskConfig.penalty_overcurl == 10.0


def test_mjx_brax_env_preserves_fixed_brax_metrics_structure():
    source = open("robot_curl_mjx/brax_env.py", encoding="utf-8").read()

    assert "metrics=reward_metrics" in source
    assert source.count('"reward_total"') >= 2
