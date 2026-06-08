import numpy as np

from env import QuadrupedFoldEnv
from policy_search import (
    CurlPolicyParams,
    FeedbackPolicyParams,
    make_action,
    make_closed_loop_action,
    make_feedback_action,
    score_row,
)


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


def test_score_row_prioritizes_more_stable_curl_over_extra_reward():
    less_curl_more_reward = {"max_curl": 0.20, "total_reward": 600.0, "done": False, "min_up": 0.99}
    more_curl_less_reward = {"max_curl": 0.26, "total_reward": 450.0, "done": False, "min_up": 0.99}

    assert score_row(more_curl_less_reward) > score_row(less_curl_more_reward)


def test_closed_loop_policy_reduces_curl_action_near_goal():
    env = QuadrupedFoldEnv()
    env.reset(seed=6)
    for _ in range(50):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    params = CurlPolicyParams(-0.03, -0.015, 0.025, -0.025, 0.0, 40, 160)

    env.data.qpos[env.qpos_addr["torso_hinge"]] = -0.05
    early = make_closed_loop_action(env, params, 30)

    env.data.qpos[env.qpos_addr["torso_hinge"]] = -env.curl_goal
    near_goal = make_closed_loop_action(env, params, 30)

    assert early[0] < near_goal[0]
    assert near_goal[0] == 0.0


def test_closed_loop_policy_is_conservative_with_low_contacts():
    env = QuadrupedFoldEnv()
    env.reset(seed=7)
    params = CurlPolicyParams(-0.03, -0.015, 0.025, -0.025, 0.0, 40, 160)

    env.foot_geom_ids = []
    action = make_closed_loop_action(env, params, 30)

    assert action[0] == 0.0


def test_feedback_policy_scales_torso_action_with_curl_error():
    env = QuadrupedFoldEnv()
    env.reset(seed=8)
    for _ in range(50):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    params = FeedbackPolicyParams(
        torso_gain=-0.05,
        front_hip_gain=-0.02,
        front_knee_gain=0.04,
        hind_hip_gain=-0.02,
        hind_knee_gain=0.0,
        phase_split=0.5,
        min_contacts=2.0,
    )

    env.data.qpos[env.qpos_addr["torso_hinge"]] = -0.05
    early = make_feedback_action(env, params)

    env.data.qpos[env.qpos_addr["torso_hinge"]] = -0.40
    late = make_feedback_action(env, params)

    assert early[0] < late[0] < 0.0


def test_feedback_policy_changes_leg_phase_by_curl_amount():
    env = QuadrupedFoldEnv()
    env.reset(seed=9)
    for _ in range(50):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    params = FeedbackPolicyParams(-0.05, -0.02, 0.04, -0.02, 0.0, 0.5, 2.0)

    env.data.qpos[env.qpos_addr["torso_hinge"]] = -0.05
    early = make_feedback_action(env, params)

    env.data.qpos[env.qpos_addr["torso_hinge"]] = -0.30
    late = make_feedback_action(env, params)

    assert early[3] > 0.0
    assert late[3] < 0.0
