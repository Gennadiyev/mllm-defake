import glob
import json
import subprocess
import sys
from functools import partial

import yaml
from trl import TrlParser, get_peft_config

from mllm_defake.finetune.configs import DEEPSPEED_SETTINGS
from mllm_defake.finetune.trainers.grpo import VLGRPOConfig, VLGRPOModelConfig, VLGRPOScriptArguments, get_jsonl_dataset
from mllm_defake.finetune.utils import get_torchrun_args, SPECIAL_TOKNES


def grpo_train(config):
    # process config
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    cmd = ["-m", "mllm_defake.finetune.grpo"]
    for key, value in config.items():
        # special keys
        if key == "torch_dtype":
            if value == "bfloat16":
                cmd.append("--bf16")
            elif value == "float16":
                cmd.append("--fp16")
        if key == "deepspeed":
            value = DEEPSPEED_SETTINGS[value]
        # append
        cmd.append(f"--{key}")
        if isinstance(value, dict):
            cmd.append(json.dumps(value))
        else:
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


def grpo_main(script_args, grpo_args, model_args):
    # reward cls
    if script_args.reward_version is None:
        raise ValueError("reward_version is required")
    if script_args.reward_version == "v0":
        from mllm_defake.finetune.rewards import RewardV0

        reward_cls = RewardV0
    else:
        raise ValueError(f"Unknown reward version: {script_args.reward_version}")
    reward_config = json.loads(script_args.reward_config) if script_args.reward_config is not None else {}
    # model and trainer
    temp_model_name_or_path = model_args.model_name_or_path.lower()
    if "qwen2.5-vl" in temp_model_name_or_path:
        from mllm_defake.finetune.trainers.grpo import GRPOTrainer_Qwen2_5_VL

        trainer_cls = partial(
            GRPOTrainer_Qwen2_5_VL,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
        )
        is_norm = False
        special_tokens = SPECIAL_TOKNES["qwen2.5-vl"]
    elif "internvl2_5" in temp_model_name_or_path:
        from mllm_defake.finetune.trainers.grpo import GRPOTrainer_InternVL2_5

        trainer_cls = GRPOTrainer_InternVL2_5
        is_norm = True
        special_tokens = SPECIAL_TOKNES["internvl2_5"]
    else:
        raise ValueError(f"Unknown model: {model_args.model_name_or_path}")
    # dataset
    dataset = get_jsonl_dataset(script_args.dataset_name, script_args.images_root, special_tokens, is_norm)
    splits = {"train": dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(test_size=script_args.val_split_ratio)
        splits["train"] = train_val_split["train"]
        splits["test"] = train_val_split["test"]
    # train
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_cls=reward_cls,
        reward_config=reward_config,
        args=grpo_args,
        train_dataset=splits["train"],
        test_dataset=splits.get("validation") if grpo_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision=model_args.freeze_vision,
    )
    # resume
    if list(glob.glob("checkpoint-*", root_dir=grpo_args.output_dir)):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # save
    trainer.save_model(grpo_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((VLGRPOScriptArguments, VLGRPOConfig, VLGRPOModelConfig))
    script_args, grpo_args, model_args = parser.parse_args_and_config()
    grpo_main(script_args, grpo_args, model_args)
