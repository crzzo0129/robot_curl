"""Command-line helpers for task configuration."""

from robot_curl.task_config import CurlTaskConfig


def add_task_config_args(parser):
    parser.add_argument("--curl-goal", type=float, default=CurlTaskConfig.curl_goal)
    parser.add_argument("--max-episode-steps", type=int, default=CurlTaskConfig.max_episode_steps)
    parser.add_argument("--action-repeat", type=int, default=CurlTaskConfig.action_repeat)
    parser.add_argument("--action-scale", type=float, default=CurlTaskConfig.action_scale)
    parser.add_argument("--reward-curl", type=float, default=CurlTaskConfig.reward_curl)
    parser.add_argument("--reward-progress", type=float, default=CurlTaskConfig.reward_progress)
    parser.add_argument("--reward-tier", type=float, default=CurlTaskConfig.reward_tier)
    parser.add_argument("--reward-contact", type=float, default=CurlTaskConfig.reward_contact)
    parser.add_argument("--reward-low-contact", type=float, default=CurlTaskConfig.reward_low_contact)
    parser.add_argument("--reward-smooth", type=float, default=CurlTaskConfig.reward_smooth)
    parser.add_argument("--reward-stable", type=float, default=CurlTaskConfig.reward_stable)
    parser.add_argument("--reward-upright", type=float, default=CurlTaskConfig.reward_upright)
    parser.add_argument("--reward-alive", type=float, default=CurlTaskConfig.reward_alive)
    parser.add_argument("--penalty-overcurl", type=float, default=CurlTaskConfig.penalty_overcurl)


def task_config_from_args(args):
    return CurlTaskConfig(
        curl_goal=args.curl_goal,
        max_episode_steps=args.max_episode_steps,
        action_repeat=args.action_repeat,
        action_scale=args.action_scale,
        reward_curl=args.reward_curl,
        reward_progress=args.reward_progress,
        reward_tier=args.reward_tier,
        reward_contact=args.reward_contact,
        reward_low_contact=args.reward_low_contact,
        reward_smooth=args.reward_smooth,
        reward_stable=args.reward_stable,
        reward_upright=args.reward_upright,
        reward_alive=args.reward_alive,
        penalty_overcurl=args.penalty_overcurl,
    )
