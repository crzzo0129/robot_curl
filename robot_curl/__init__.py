"""Robot curl simulation package."""
from robot_curl.task_config import CurlTaskConfig, JOINT_NAMES, N_JOINTS

__all__ = ["CurlTaskConfig", "JOINT_NAMES", "N_JOINTS", "QuadrupedFoldEnv"]


def __getattr__(name):
    if name == "QuadrupedFoldEnv":
        from robot_curl.env import QuadrupedFoldEnv

        return QuadrupedFoldEnv
    raise AttributeError(name)
