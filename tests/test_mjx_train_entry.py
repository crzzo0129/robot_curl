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
    assert args.action_repeat == 1
    assert args.hidden_layers == [256, 128, 128, 128]
    assert args.activation == "elu"
    assert args.train_policy_videos is False
    assert args.final_policy_video is True


def test_mjx_pipeline_parses_hidden_layers():
    from robot_curl_mjx.pipeline import hidden_layers_tuple

    assert hidden_layers_tuple([64, 32]) == (64, 32)


def test_mjx_pipeline_configures_cloud_runtime(monkeypatch):
    from robot_curl_mjx.pipeline import configure_cloud_runtime

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "")
    configure_cloud_runtime(xla_triton=True, mujoco_gl="osmesa")

    assert "xla_gpu_triton_gemm_any=True" in __import__("os").environ["XLA_FLAGS"]
    assert __import__("os").environ["MUJOCO_GL"] == "osmesa"


def test_mjx_playback_defaults_use_osmesa_video_path():
    from scripts.mjx_playback import parse_args

    args = parse_args([])

    assert args.episodes == 3
    assert args.episode_length == 128
    assert args.params == "mjx_runs/curl_smoke/params"
    assert args.video == "mjx_runs/curl_smoke/playback.mp4"


def test_mjx_brax_env_factory_imports_without_heavy_dependencies():
    from robot_curl_mjx.brax_env import make_brax_env

    assert callable(make_brax_env)


def test_mjx_brax_env_preserves_brax_metrics_structure():
    source = open("robot_curl_mjx/brax_env.py", encoding="utf-8").read()

    assert "metrics=state.metrics" in source
