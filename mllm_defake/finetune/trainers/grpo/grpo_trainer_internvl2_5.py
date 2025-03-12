import PIL
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
)
from trl.data_utils import apply_chat_template

from mllm_defake.finetune.trainers.grpo.base_grpo_trainer import BaseGRPOTrainer


class GRPOTrainer_InternVL2_5(BaseGRPOTrainer):
    """The trainer for GRPO with InternVL2.5 model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    def _build_model(
        self, model_name_or_path: str, model_init_kwargs: dict
    ) -> tuple[PreTrainedModel, list[str], AutoTokenizer, int, type]:
        """Build the model, specify the vision modules, and return the tokenizer and pad token id."""
        # model
        model = AutoModel.from_pretrained(model_name_or_path, trust_model=True, **model_init_kwargs)
        # vision modules
        vision_modules = ["vision"]
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_model=True)
        pad_token_id = tokenizer.pad_token_id
        return model, vision_modules, tokenizer, pad_token_id, AutoModel

    def _process_input(self, inputs: list[dict]) -> dict:
        conversation_contents = [apply_chat_template(x, self.processing_class)["prompt"] for x in inputs]
        # handle both pre-loaded images and image paths
        images = []
        for x in inputs:
            if "image" in x:
                img = x["image"]
            elif "image_path" in x and x["image_path"] is not None:
                img = PIL.Image.open(x["image_path"]).convert("RGB")
            images.append(img)
        pixel_values = [self._load_image(image) for image in images]
        num_patches_list = [p.size(0) for p in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        # process query
        queries = []
        for i, num_patches in enumerate(num_patches_list):
            conversation_content = conversation_contents[i]
            image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * num_patches + self.IMG_END_TOKEN
            query = conversation_content.replace("<image>", image_tokens)
            queries.append(query)
        prompt_inputs = self.processing_class(
            texts=queries,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs["pixel_values"] = pixel_values
        return prompt_inputs

    def _build_transforms(self, input_size: int):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_image(self, image, input_size=448, max_num=12):
        transform = self._build_transforms(input_size=input_size)
        images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
