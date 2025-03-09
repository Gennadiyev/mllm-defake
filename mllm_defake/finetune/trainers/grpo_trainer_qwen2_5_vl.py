from transformers import (
    AutoProcessor,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration
)

from .base_grpo_trainer import BaseGRPOTrainer


class GRPOTrainer_Qwen2_5_VL(BaseGRPOTrainer):
    """The trainer for GRPO with Qwen2.5-VL model."""

    def __init__(self, *args, **kwargs):
        self.min_pixels = kwargs.pop("min_pixels", 3136)
        self.max_pixels = kwargs.pop("max_pixels", 12845056)
        super().__init__(*args, **kwargs)
    
    def _build_model(
        self, model_name_or_path: str, model_init_kwargs: dict
    ) -> tuple[PreTrainedModel, list[str], AutoProcessor, int]:
        """Build the model, specify the vision modules, and return the processor or tokenizer and pad token id."""
        # model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_init_kwargs)
        # vision modules
        vision_modules = ["visual"]
        # processor
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        pad_token_id = processor.tokenizer.pad_token_id
        processor.pad_token_id = pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id
        # min/max pixels
        processor.image_processor.min_pixels = self.min_pixels
        processor.image_processor.max_pixels = self.max_pixels
        return model, vision_modules, processor, pad_token_id
