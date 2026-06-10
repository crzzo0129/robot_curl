"""MJX backend for robot curl experiments."""

__all__ = ["QuadrupedCurlMJXEnv"]


def __getattr__(name):
    if name == "QuadrupedCurlMJXEnv":
        from robot_curl_mjx.env import QuadrupedCurlMJXEnv

        return QuadrupedCurlMJXEnv
    raise AttributeError(name)
