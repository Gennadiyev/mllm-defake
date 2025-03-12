import PIL
from transformers import AutoProcessor, PreTrainedModel, Qwen2_5_VLForConditionalGeneration
from trl.data_utils import apply_chat_template

from mllm_defake.finetune.trainers.grpo.base_grpo_trainer import BaseGRPOTrainer


class GRPOTrainer_Qwen2_5_VL(BaseGRPOTrainer):
    """The trainer for GRPO with Qwen2.5-VL model."""

    def __init__(self, *args, **kwargs):
        self.min_pixels = kwargs.pop("min_pixels", 3136)
        self.max_pixels = kwargs.pop("max_pixels", 12845056)
        super().__init__(*args, **kwargs)

    def _build_model(
        self, model_name_or_path: str, model_init_kwargs: dict
    ) -> tuple[PreTrainedModel, list[str], AutoProcessor, int, type]:
        """Build the model, specify the vision modules, and return the processor and pad token id."""
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
        return model, vision_modules, processor, pad_token_id, Qwen2_5_VLForConditionalGeneration

    def _process_input(self, inputs: list[dict]) -> dict:
        # add image to inputs
        new_inputs = []
        for x in inputs:
            prompt = x["prompt"]
            last_content = prompt[-1]["content"]
            prompt[-1]["content"] = [
                {'type': 'image', 'text': None},
                {'type': 'text', 'text': last_content}
            ]
            new_inputs.append({"prompt": prompt})
        conversation_contents = [apply_chat_template(x, self.processing_class)["prompt"] for x in new_inputs]
        # replace <image>
        conversation_contents = [x.replace("<image>", "") for x in conversation_contents]
        # handle both pre-loaded images and image paths
        images = []
        for x in inputs:
            if "image" in x:
                img = x["image"]
            elif "image_path" in x and x["image_path"] is not None:
                img = PIL.Image.open(x["image_path"])

            # ensure minimum dimensions of 28 pixels
            w, h = img.size
            if w < 28 or h < 28:
                # calculate new dimensions maintaining aspect ratio
                if w < h:
                    new_w = 28
                    new_h = int(h * (28 / w))
                else:
                    new_h = 28
                    new_w = int(w * (28 / h))
                img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)

            images.append(img)

        if len(images) > 0:
            prompt_inputs = self.processing_class(
                text=conversation_contents,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        else:
            prompt_inputs = self.processing_class(
                text=conversation_contents,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        return prompt_inputs
