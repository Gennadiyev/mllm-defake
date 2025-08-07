import subprocess
import sys

import yaml
from swift.llm.train import SwiftRLHF

import mllm_defake.finetune.rewards
from mllm_defake.finetune.utils import _get_torchrun_args


def grpo_train(config):
    # process config
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    # check if rlhf_type is provided
    if 'rlhf_type' not in config:
        raise ValueError("rlhf_type must be provided in the config file")

    cmd = ["-m", "mllm_defake.finetune.grpo"]
    for key, value in config.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    # add external_plugins
    cmd.append("--external_plugins")
    cmd.append(mllm_defake.finetune.rewards.__file__)

    # hardcode to disable versioning
    cmd.append("--no_add_version")
    # run
    torchrun_args = _get_torchrun_args()
    if torchrun_args is None:
        cmd = ["python", *cmd]
    else:
        cmd = ["torchrun", *torchrun_args, *cmd]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def grpo_main(args=None):
    return SwiftRLHF(args).main()


if __name__ == "__main__":
    grpo_main()
