from argparse import ArgumentParser

from robot_curl.config_args import add_task_config_args, task_config_from_args
from robot_curl.wandb_utils import add_wandb_args, build_wandb_callback, finish_wandb_run, init_wandb_run


class FakeWandb:
    def __init__(self):
        self.calls = []
        self.finished = False

    def init(self, **kwargs):
        self.calls.append(kwargs)
        return object()

    def finish(self):
        self.finished = True


def _parser():
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=123)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--envs", type=int, default=2)
    add_task_config_args(parser)
    add_wandb_args(parser)
    return parser


def test_wandb_is_disabled_by_default():
    args = _parser().parse_args([])

    assert init_wandb_run(args, task_config_from_args(args), script_name="quick_train") is None
    assert build_wandb_callback(args) is None


def test_wandb_init_uses_training_and_task_config():
    args = _parser().parse_args(["--wandb", "--wandb-project", "robot-curl-test", "--wandb-name", "smoke"])
    fake = FakeWandb()

    run = init_wandb_run(args, task_config_from_args(args), script_name="quick_train", wandb_module=fake)

    assert run is not None
    assert fake.calls[0]["project"] == "robot-curl-test"
    assert fake.calls[0]["name"] == "smoke"
    assert fake.calls[0]["sync_tensorboard"] is True
    assert fake.calls[0]["config"]["script"] == "quick_train"
    assert fake.calls[0]["config"]["curl_goal"] == args.curl_goal
    assert fake.calls[0]["config"]["steps"] == 123


def test_finish_wandb_run_only_finishes_enabled_runs():
    fake = FakeWandb()

    finish_wandb_run(run=object(), wandb_module=fake)

    assert fake.finished
