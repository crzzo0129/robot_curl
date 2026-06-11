"""Reward helpers shared by the MJX backend."""
import numpy as np


def curl_reward_terms(
    config,
    curl,
    init_curl,
    best_curl,
    contact_count,
    action_sq_mean,
    torso_vel_sq,
    torso_angvel_sq,
    upright,
    leg_error,
):
    effective_curl = min(curl, config.curl_goal)
    curl_progress = max(0.0, effective_curl - init_curl)
    overcurl = max(0.0, curl - config.curl_goal)

    r_curl = config.reward_curl * min(curl, config.curl_goal) / max(config.curl_goal, 1e-6)
    r_progress = config.reward_progress * curl_progress

    r_tier = 0.0
    for threshold in config.curl_tiers:
        if curl > threshold and best_curl <= threshold:
            r_tier += config.reward_tier

    r_contact = config.reward_contact * min(contact_count, config.contact_cap)
    r_contact -= config.reward_low_contact * max(0.0, config.min_contacts - contact_count)

    r_smooth = -action_sq_mean
    r_stable = -0.5 * torso_vel_sq - 0.2 * torso_angvel_sq
    r_upright = -config.reward_upright * max(0.0, config.upright_threshold - upright)
    r_leg_fold = config.reward_leg_fold * np.exp(-4.0 * leg_error)

    reward = (
        r_curl
        + r_progress
        + r_tier
        + r_contact
        + config.reward_smooth * r_smooth
        + config.reward_stable * r_stable
        + r_upright
        + r_leg_fold
        + config.reward_alive
        - config.penalty_overcurl * overcurl
    )
    return reward, max(best_curl, curl)
