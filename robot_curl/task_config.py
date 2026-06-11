"""Lightweight task configuration shared by CPU and MJX environments."""
from dataclasses import dataclass


JOINT_NAMES = [
    "torso_hinge",
    "fl_hip_flex", "fl_knee",
    "fr_hip_flex", "fr_knee",
    "hl_hip_flex", "hl_knee",
    "hr_hip_flex", "hr_knee",
]
N_JOINTS = len(JOINT_NAMES)


@dataclass(frozen=True)
class CurlTaskConfig:
    curl_goal: float = 0.45
    max_episode_steps: int = 250
    action_repeat: int = 20
    action_scale: float = 0.1
    reward_curl: float = 3.0
    reward_progress: float = 2.0
    reward_tier: float = 0.5
    curl_tiers: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    reward_contact: float = 0.15
    reward_low_contact: float = 0.25
    min_contacts: float = 2.0
    contact_cap: float = 3.0
    reward_smooth: float = 0.03
    reward_stable: float = 0.02
    reward_upright: float = 4.0
    upright_threshold: float = 0.9
    reward_alive: float = 0.05
    reward_leg_fold: float = 0.5
    penalty_overcurl: float = 10.0
    terminate_upright: float = 0.3
    terminate_height: float = 0.05
