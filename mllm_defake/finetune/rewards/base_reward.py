from swift.plugin import ORM

from mllm_defake.finetune.rewards.utils import cal_iou, check_format, check_label


class BaseReward(ORM):
    
    # reward base, can be used to shift the reward distribution
    reward_base = 0
    # iout reward
    iou_reward_weight = 1.0
    iou_reward_exponent = 1.0
    # label reward
    label_correct_reward_weight = 1.0
    label_incorrect_reward_weight = -1.0
    # format reward
    format_correct_reward_weight = 1.0
    format_incorrect_reward_weight = -1.0

    def __call__(self, completions: list[str], objects: list[dict], images: list[str], **kwargs):
        """Calculate the reward for the given completions and objects.
        
        Args:
            completions (list[str]): List of completions to evaluate.
            objects (list[dict]): List of objects to evaluate against. There are two keys in each dict:
                - "ref": list[str], the reference text for the object.
                - "bbox": list[list[float]], the bounding box coordinates for the object.
            The `objects` is the same as the data format used in SFT stage.
            images (list[str]): List of image paths corresponding to the completions.
        
        Returns:
            list[float]: List of rewards for each completion.
        """
        rewards = []
        for this_completion, this_objects, this_images in zip(completions, objects, images):
            reward = self.reward_base
            # calculate iou reward
            # if this_objects is empty, iou is 1
            iou = cal_iou(this_completion, this_objects, this_images)
            iou_reward = self.iou_reward_weight * (iou ** self.iou_reward_exponent)
            reward += iou_reward
            # calculate label reward
            # if this_objects is empty, ground truth is `real`
            is_label_correct = check_label(this_completion, this_objects)
            if is_label_correct:
                label_reward = self.label_correct_reward_weight
            else:
                label_reward = self.label_incorrect_reward_weight
            reward += label_reward
            # calculate format reward
            # if this_objects is empty, grounding format is not required
            is_format_correct = check_format(this_completion, this_objects)
            if is_format_correct:
                format_reward = self.format_correct_reward_weight
            else:
                format_reward = self.format_incorrect_reward_weight
            reward += format_reward
            rewards.append(reward)
        return rewards
