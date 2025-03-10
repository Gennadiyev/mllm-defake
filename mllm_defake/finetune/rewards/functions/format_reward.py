import re


def format_reward_function(pattern: str, completion: str) -> float:
    """
    Find the target format in the completion and return the reward.
    """
    matches = re.fullmatch(pattern, completion, re.DOTALL)
    if matches:
        return 1.0
    return 0.0
