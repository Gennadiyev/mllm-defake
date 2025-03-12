import json
import os

import imagesize
from datasets import Dataset


def get_jsonl_dataset(data_file: str, images_root: str, special_tokens: dict, is_norm: bool) -> Dataset:
    data = []
    with open(data_file) as f:
        for line in f:
            this_data = json.loads(line)
            this_data = _make_item(this_data, images_root, special_tokens, is_norm)
            data.append(this_data)
    dataset = Dataset.from_list(data)
    return dataset


def _make_item(item: dict, images_root: str, special_tokens: dict, is_norm: bool) -> dict:
    images = item["images"]
    if len(images) > 1:
        raise ValueError("Only one image is supported")
    image = os.path.join(images_root, images[0]) if len(images) > 0 else None
    messages = item["messages"]
    objects = item.get("objects", None)
    # extract message
    has_system = messages[0]["role"] == "system"
    if has_system:
        user_input = messages[1]["content"]
        assistant_output = messages[2]["content"]
    else:
        user_input = messages[0]["content"]
        assistant_output = messages[1]["content"]
    # replace special tokens
    if objects is not None:
        refs = objects["ref"]
        bboxes = objects["bbox"]
        if is_norm:
            bboxes = _norm_bbox(image, bboxes)
        for ref, bbox in zip(refs, bboxes, strict=False):
            ref_object = f"{special_tokens['ref_object_start']}{ref}{special_tokens['ref_object_end']}"
            bbox_object = f"{special_tokens['box_start']}{str(bbox)}{special_tokens['box_end']}"
            assistant_output = assistant_output.replace("<ref-object>", ref_object, 1)
            assistant_output = assistant_output.replace("<bbox>", bbox_object, 1)
    # build new conversation
    conversation = []
    if has_system:
        conversation.append({"role": "system", "content": messages[0]["content"]})
    conversation.append({"role": "user", "content": user_input})
    item = {
        "image_path": image,
        "user_input": user_input.replace("<image>", ""),
        "assistant_output": assistant_output,
        "prompt": conversation,
    }
    return item


def _norm_bbox(image_path: str, bboxes: list) -> list:
    width, height = imagesize.get(image_path)
    normed_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 /= width * 1000
        y1 /= height * 1000
        x2 /= width * 1000
        y2 /= height * 1000
        normed_bboxes.append([x1, y1, x2, y2])
    return normed_bboxes
