from swift.plugin import orms

from mllm_defake.finetune.rewards.stage1_reward import Stage1Reward
from mllm_defake.finetune.rewards.stage2_reward import Stage2Reward
from mllm_defake.finetune.rewards.stage3_reward import Stage3Reward

orms["stage1_reward"] = Stage1Reward
orms["stage2_reward"] = Stage2Reward
orms["stage3_reward"] = Stage3Reward
