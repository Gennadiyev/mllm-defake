from dataclasses import dataclass, field

from trl import GRPOConfig, ModelConfig, ScriptArguments


@dataclass
class VLGRPOConfig(GRPOConfig):
    """
    Args for callbacks, benchmarks and system prompt.
    """

    benchmarks: list[str] = field(default_factory=list, metadata={"help": "The benchmarks to run after training."})
    callbacks: list[str] = field(default_factory=list, metadata={"help": "The callbacks to run during training."})
    system_prompt: str | None = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )


@dataclass
class VLGRPOModelConfig(ModelConfig):
    freeze_vision: bool = False


@dataclass
class VLGRPOScriptArguments(ScriptArguments):
    """
    script arguments for the GRPO training script.
    """

    data_file: str = field(
        default=None,
        metadata={"help": "The path to the data file in `jsonl` format."},
    )
    images_root: str = field(
        default="",
        metadata={"help": "The root directory of the images"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_version: str = field(
        default=None,
        metadata={"help": "Reward version"},
    )
    reward_config: str = field(
        default=None,
        metadata={"help": "Reward config in json format string"},
    )
    max_pixels: int | None = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: int | None = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
