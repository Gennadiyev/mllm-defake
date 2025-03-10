import json
import os

from datasets import Dataset


def get_jsonl_dataset(data_file: str, images_root: str, special_tokens: dict) -> Dataset:
    data = []
    with open(data_file) as f:
        for line in f:
            data.append(json.loads(line))
    dataset = Dataset.from_list(data)
    dataset = dataset.map(lambda x: _make_item(x, images_root, special_tokens))
    return dataset


def _make_item(item: dict, images_root: str, special_tokens: dict) -> dict:
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
    conversation.append({"role": "assistant", "content": assistant_output})
    item = {
        "image_path": image,
        "user_input": user_input,
        "assistant_output": assistant_output,
        "conversation": conversation,
    }
    return item
