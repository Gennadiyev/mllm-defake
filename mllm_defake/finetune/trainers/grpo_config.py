from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):
    pass
