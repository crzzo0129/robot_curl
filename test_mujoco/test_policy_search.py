import numpy as np

from env import QuadrupedFoldEnv
from policy_search import CurlPolicyParams, make_action, score_row


def test_make_action_uses_torso_and_leg_adjustment_windows():
    env = QuadrupedFoldEnv()
    params = CurlPolicyParams(
        torso=-0.03,
        front_hip=0.02,
        front_knee=0.04,
        hind_hip=-0.01,
        hind_knee=-0.02,
        switch_step=10,
        release_step=20,
    )

    early = make_action(env, params, 5)
    late = make_action(env, params, 25)

    assert early.shape == env.action_space.shape
    assert early[0] == params.torso
    assert early[2] == params.front_hip
    assert early[3] == params.front_knee
    assert early[8] == params.hind_hip
    assert late[0] == params.torso
    assert np.count_nonzero(late[1:]) == 0


def test_score_row_prefers_curl_without_falling():
    stable_curled = {"max_curl": 0.25, "total_reward": 20.0, "done": False, "min_up": 0.95}
    fallen_curled = {"max_curl": 0.30, "total_reward": 30.0, "done": True, "min_up": 0.20}
    stable_flat = {"max_curl": 0.02, "total_reward": 50.0, "done": False, "min_up": 1.0}

    assert score_row(stable_curled) > score_row(fallen_curled)
    assert score_row(stable_curled) > score_row(stable_flat)
