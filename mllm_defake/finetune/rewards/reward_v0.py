from mllm_defake.finetune.rewards.base_reward import BaseReward
from mllm_defake.finetune.rewards.functions import format_reward_function, verdict_reward_function


class RewardV0(BaseReward):
    def __init__(self):
        self.num_functions = 2
        self.reward_names = ["format", "verdict"]
        super().__init__()

    def __call__(self, user_input: str, assistant_output: str, completion: str) -> dict:
        pattern = r"<think>.*?</think>.*?<verdict>.*?</verdict>"
        format_reward = format_reward_function(pattern, completion)
        verdict_reward = verdict_reward_function(assistant_output, completion)
        all_reward = format_reward + verdict_reward
        return {"all_reward": all_reward, "per_function_reward": [format_reward, verdict_reward]}
