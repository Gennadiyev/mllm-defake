from mllm_defake.finetune.rewards.base_reward import BaseReward


class Stage1Reward(BaseReward):

    # reward base, can be used to shift the reward distribution
    reward_base = 0
    # iout reward
    iou_reward_weight = 1.0
    iou_reward_exponent = 1.1
    # label reward
    label_correct_reward_weight = 1.0
    label_incorrect_reward_weight = -1.0
    # format reward
    format_correct_reward_weight = 2.0
    format_incorrect_reward_weight = -1.0
