import re


def verdict_reward_function(assistant_output: str, completion: str) -> float:
    pattern = r"<verdict>(.*?)</verdict>"
    ground_truth = re.search(pattern, assistant_output)
    if ground_truth:
        ground_truth = ground_truth.group(1)
    else:
        raise ValueError("No verdict found in the ground truth.")
    pred = re.search(pattern, completion)
    if pred:
        pred = pred.group(1)
    else:
        return 0.0
    return 1.0 if ground_truth == pred else 0.5
