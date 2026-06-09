def test_mjx_smoke_test_module_exposes_main():
    from robot_curl_mjx import smoke_test

    assert callable(smoke_test.main)


def test_mjx_environment_class_imports_without_initializing_gpu():
    from robot_curl_mjx.env import QuadrupedCurlMJXEnv

    assert QuadrupedCurlMJXEnv.__name__ == "QuadrupedCurlMJXEnv"


def test_mjx_smoke_defaults_are_lightweight():
    from robot_curl_mjx.smoke_test import parse_args

    args = parse_args([])

    assert args.action_repeat == 1
    assert args.settle_steps == 0
    assert args.max_episode_steps == 10
