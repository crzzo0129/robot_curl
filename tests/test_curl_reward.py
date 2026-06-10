import numpy as np

from robot_curl.env import CurlTaskConfig, JOINT_NAMES, QuadrupedFoldEnv


def _set_joint(env, name, value):
    env.data.qpos[env.qpos_addr[name]] = value


def test_torso_curl_progress_is_rewarded_without_leg_targets():
    env = QuadrupedFoldEnv()
    env.reset(seed=1)

    base_reward = env._compute_reward(np.zeros(env.action_space.shape, dtype=np.float32))

    _set_joint(env, "torso_hinge", 0.35)
    changed_legs = [
        name
        for name in JOINT_NAMES
        if name != "torso_hinge" and not np.isclose(env.data.qpos[env.qpos_addr[name]], env.fold_pose[JOINT_NAMES.index(name)])
    ]
    assert changed_legs

    curled_reward = env._compute_reward(np.zeros(env.action_space.shape, dtype=np.float32))

    assert curled_reward > base_reward


def test_task_config_controls_goal_episode_length_and_action_scale():
    env = QuadrupedFoldEnv(config=CurlTaskConfig(curl_goal=0.20, max_episode_steps=42, action_scale=0.05))

    assert env.curl_goal == 0.20
    assert env.max_episode_steps == 42
    assert np.allclose(env.action_space.low, -0.05)
    assert np.allclose(env.action_space.high, 0.05)


def test_reward_weights_are_configurable():
    quiet = CurlTaskConfig(
        reward_curl=0.0,
        reward_progress=0.0,
        reward_tier=0.0,
        reward_contact=0.0,
        reward_low_contact=0.0,
        reward_smooth=0.0,
        reward_stable=0.0,
        reward_upright=0.0,
        reward_alive=0.0,
        penalty_overcurl=0.0,
    )
    curl_rewarded = CurlTaskConfig(
        reward_curl=5.0,
        reward_progress=0.0,
        reward_tier=0.0,
        reward_contact=0.0,
        reward_low_contact=0.0,
        reward_smooth=0.0,
        reward_stable=0.0,
        reward_upright=0.0,
        reward_alive=0.0,
        penalty_overcurl=0.0,
    )
    quiet_env = QuadrupedFoldEnv(config=quiet)
    rewarded_env = QuadrupedFoldEnv(config=curl_rewarded)
    quiet_env.reset(seed=10)
    rewarded_env.reset(seed=10)

    _set_joint(quiet_env, "torso_hinge", 0.20)
    _set_joint(rewarded_env, "torso_hinge", 0.20)

    assert rewarded_env._compute_reward(np.zeros(rewarded_env.action_space.shape, dtype=np.float32)) > quiet_env._compute_reward(
        np.zeros(quiet_env.action_space.shape, dtype=np.float32)
    )


def test_leg_only_fold_pose_is_not_the_task():
    env = QuadrupedFoldEnv()
    env.reset(seed=3)

    base_reward = env._compute_reward(np.zeros(env.action_space.shape, dtype=np.float32))

    for i, name in enumerate(JOINT_NAMES):
        if name != "torso_hinge":
            _set_joint(env, name, env.fold_pose[i])

    leg_only_reward = env._compute_reward(np.zeros(env.action_space.shape, dtype=np.float32))

    assert leg_only_reward <= base_reward


def test_foot_contact_observation_reports_named_feet():
    env = QuadrupedFoldEnv()
    obs, _ = env.reset(seed=2)
    for _ in range(50):
        obs, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not terminated
        assert not truncated

    contacts = obs[-4:]

    assert contacts.shape == (4,)
    assert np.all(contacts == 1.0)


def test_joint_torques_are_limited():
    env = QuadrupedFoldEnv()
    env.reset(seed=4)

    env.target_q[:] = env.jnt_high
    env._apply_pd()

    for i, name in enumerate(JOINT_NAMES):
        torque = env.data.qfrc_applied[env.dof_addr[name]]
        assert abs(torque) <= env.torque_limits[i] + 1e-6


def test_overcurl_is_not_more_rewarding_than_goal_curl():
    env = QuadrupedFoldEnv()
    env.reset(seed=5)

    _set_joint(env, "torso_hinge", env.curl_goal)
    goal_reward = env._compute_reward(np.zeros(env.action_space.shape, dtype=np.float32))

    _set_joint(env, "torso_hinge", env.curl_goal + 0.5)
    overcurl_reward = env._compute_reward(np.zeros(env.action_space.shape, dtype=np.float32))

    assert overcurl_reward <= goal_reward


def test_positive_torso_hinge_is_the_curl_direction():
    env = QuadrupedFoldEnv()
    env.reset(seed=6)

    _set_joint(env, "torso_hinge", 0.35)
    assert np.isclose(env._curl_amount(), 0.35)

    _set_joint(env, "torso_hinge", -0.35)
    assert np.isclose(env._curl_amount(), 0.0)
