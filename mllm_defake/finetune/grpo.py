from dataclasses import dataclass

from trl import ModelConfig, ScriptArguments, TrlParser

from mllm_defake.finetune.trainers.grpo_config import GRPOConfig


@dataclass
class GRPOScriptArguments(ScriptArguments):
    pass


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def grpo_train(config):
    pass


def grpo_main(script_args, grpo_config, model_config):
    pass


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, grpo_config, model_config = parser.parse_args_and_config()
    grpo_main(script_args, grpo_config, model_config)
