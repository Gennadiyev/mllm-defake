import subprocess
import sys

import yaml

from mllm_defake.finetune.utils import get_torchrun_args


def sft_train(config):
    # process config
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    cmd = ["-m", "mllm_defake.finetune.trainers.swift_sft_trainer"]
    for key, value in config.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    # run
    torchrun_args = get_torchrun_args()
    if torchrun_args is None:
        cmd = ["python", *cmd]
    else:
        cmd = ["torchrun", *torchrun_args, *cmd]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
