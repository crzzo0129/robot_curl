def test_mjx_smoke_test_module_exposes_main():
    from robot_curl_mjx import smoke_test

    assert callable(smoke_test.main)


def test_mjx_environment_class_imports_without_initializing_gpu():
    from robot_curl_mjx.env import QuadrupedCurlMJXEnv

    assert QuadrupedCurlMJXEnv.__name__ == "QuadrupedCurlMJXEnv"
