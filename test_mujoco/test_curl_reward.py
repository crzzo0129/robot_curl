import numpy as np

from env import JOINT_NAMES, QuadrupedFoldEnv


def _set_joint(env, name, value):
    env.data.qpos[env.qpos_addr[name]] = value


def test_torso_curl_progress_is_rewarded_without_leg_targets():
    env = QuadrupedFoldEnv()
    env.reset(seed=1)

    base_reward = env._compute_reward(np.zeros(env.action_space.shape, dtype=np.float32))

    _set_joint(env, "torso_hinge", -0.35)
    changed_legs = [
        name
        for name in JOINT_NAMES
        if name != "torso_hinge" and not np.isclose(env.data.qpos[env.qpos_addr[name]], env.fold_pose[JOINT_NAMES.index(name)])
    ]
    assert changed_legs

    curled_reward = env._compute_reward(np.zeros(env.action_space.shape, dtype=np.float32))

    assert curled_reward > base_reward


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
