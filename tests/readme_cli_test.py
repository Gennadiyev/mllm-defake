"""
This file tests all commands listed in the README.md file.
"""

import os
import subprocess
from pathlib import Path

# Path to the root directory of the project
ROOT_DIR = Path(__file__).parent.parent
COMMANDS = {
    "classify": ["mllmdf classify demo/real/img118131.jpg --model gpt4omini"],
    "infer": [
        "mllmdf infer --model gpt4omini --real_dir demo/real --fake_dir demo/fake --count 1",
        "mllmdf doc",
    ],
}


def test_classify():
    """
    Tests all commands listed in the README.md file.
    """
    commands = COMMANDS["classify"]
    for command in commands:
        result = subprocess.run(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT_DIR,
            env=os.environ.copy(),
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()


def test_infer():
    """
    Tests all commands listed in the README.md file.
    """
    commands = COMMANDS["infer"]
    for command in commands:
        result = subprocess.run(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT_DIR,
            env=os.environ.copy(),
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
