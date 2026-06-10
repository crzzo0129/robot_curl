def test_mjx_train_entry_imports_without_initializing_gpu():
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
    assert args.num_eval_envs == 128
    assert args.unroll_length == 20
    assert args.num_minibatches == 32
    assert args.action_repeat == 1
    assert args.hidden_layers == [256, 128, 128, 128]
    assert args.activation == "elu"
    assert args.mujoco_gl == "auto"
    assert args.matmul_precision == "high"
    assert args.runtime_diagnostics is True
    assert not hasattr(args, "train_policy_videos")
    assert args.final_policy_video is True


def test_mjx_progress_logs_with_environment_step():
    from scripts.mjx_train import _make_progress_fn

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step=None):
            self.logged.append((metrics, step))

    run = FakeRun()
    progress_times = []
    progress = _make_progress_fn(run, progress_times)

    progress(51200, {"eval/episode_reward": 12.5})

    assert progress_times[0][0] == 51200
    assert run.logged[0][1] == 51200
    assert run.logged[0][0]["train_step"] == 51200
    assert run.logged[0][0]["eval/episode_reward"] == 12.5


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
    assert args.mujoco_gl == "auto"
    assert not hasattr(args, "wandb")


def test_mjx_training_policy_callback_is_callable_noop():
    from scripts.mjx_train import _noop_policy_params_fn

    assert _noop_policy_params_fn(100, object(), object()) is None


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


def test_mjx_brax_env_preserves_brax_metrics_structure():
    source = open("robot_curl_mjx/brax_env.py", encoding="utf-8").read()

    assert "metrics=state.metrics" in source
