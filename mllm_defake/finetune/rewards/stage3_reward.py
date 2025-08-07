from mllm_defake.finetune.rewards.base_reward import BaseReward


class Stage3Reward(BaseReward):

    # reward base, can be used to shift the reward distribution
    reward_base = -1
    # iout reward
    iou_reward_weight = 2.0
    iou_reward_exponent = 1.1
    # label reward
    label_correct_reward_weight = 0.5
    label_incorrect_reward_weight = -1.0
    # format reward
    format_correct_reward_weight = 0.5
    format_incorrect_reward_weight = -1.0
