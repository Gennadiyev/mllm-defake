import subprocess
import sys

import yaml
from swift.llm.train import SwiftSft

from mllm_defake.finetune.utils import _get_torchrun_args


def sft_train(config):
    # process config
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    cmd = ["-m", "mllm_defake.finetune.sft"]
    for key, value in config.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
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


def sft_main(args=None):
    return SwiftSft(args).main()


if __name__ == "__main__":
    sft_main()
