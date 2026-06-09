def test_mjx_train_entry_imports_without_initializing_gpu():
    from scripts import mjx_train

    assert callable(mjx_train.main)


def test_mjx_train_defaults_are_gpu_smoke_sized():
    from scripts.mjx_train import parse_args

    args = parse_args([])

    assert args.steps == 10_000
    assert args.envs == 128
    assert args.episode_length == 128
    assert args.action_repeat == 1


def test_mjx_brax_env_factory_imports_without_heavy_dependencies():
    from robot_curl_mjx.brax_env import make_brax_env

    assert callable(make_brax_env)
