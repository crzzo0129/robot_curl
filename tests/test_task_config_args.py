from argparse import ArgumentParser

from robot_curl.config_args import add_task_config_args, task_config_from_args


def test_task_config_args_override_environment_defaults():
    parser = ArgumentParser()
    add_task_config_args(parser)

    args = parser.parse_args(
        [
            "--curl-goal",
            "0.25",
            "--max-episode-steps",
            "100",
            "--action-scale",
            "0.05",
            "--reward-curl",
            "4.0",
            "--reward-progress",
            "1.5",
        ]
    )
    config = task_config_from_args(args)

    assert config.curl_goal == 0.25
    assert config.max_episode_steps == 100
    assert config.action_scale == 0.05
    assert config.reward_curl == 4.0
    assert config.reward_progress == 1.5
