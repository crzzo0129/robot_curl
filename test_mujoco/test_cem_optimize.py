import numpy as np

from cem_optimize import FEEDBACK_PARAM_BOUNDS, PARAM_BOUNDS, feedback_params_from_vector, params_from_vector, update_distribution


def test_params_from_vector_clips_to_valid_policy_ranges():
    vector = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, -100.0, 1000.0])
    params = params_from_vector(vector)

    for value, (low, high) in zip(
        [
            params.torso,
            params.front_hip,
            params.front_knee,
            params.hind_hip,
            params.hind_knee,
            params.switch_step,
            params.release_step,
        ],
        PARAM_BOUNDS,
    ):
        assert low <= value <= high
    assert isinstance(params.switch_step, int)
    assert isinstance(params.release_step, int)


def test_update_distribution_moves_toward_elite_samples():
    samples = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=float,
    )
    scores = np.array([0.0, 1.0, 10.0, 11.0])

    mean, std = update_distribution(samples, scores, elite_frac=0.5)

    assert np.allclose(mean, np.array([2.5, 2.5]))
    assert np.all(std > 0)


def test_feedback_params_from_vector_clips_to_valid_ranges():
    vector = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, -5.0, 10.0])
    params = feedback_params_from_vector(vector)

    values = [
        params.torso_gain,
        params.front_hip_gain,
        params.front_knee_gain,
        params.hind_hip_gain,
        params.hind_knee_gain,
        params.phase_split,
        params.min_contacts,
    ]
    for value, (low, high) in zip(values, FEEDBACK_PARAM_BOUNDS):
        assert low <= value <= high
