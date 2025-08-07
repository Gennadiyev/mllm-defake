import json
import math
import re

import imagesize


SPECIAL_TOKENS = {
    "internvl3": {
        "ref_object_start": "<ref>",
        "ref_object_end": "</ref>",
        "box_start": "<box>",
        "box_end": "</box>",
    },
    "qwen2.5-vl": {
        "ref_object_start": "<|object_ref_start|>",
        "ref_object_end": "<|object_ref_end|>",
        "box_start": "<|box_start|>",
        "box_end": "<|box_end|>",
    },
}


def _get_model_type(completion: str) -> str:
    """Determine the model type based on the completion text.

    Args:
        completion (str): The completion text.

    Returns:
        str: The model type, either "internvl3" or "qwen2.5-vl".
    """
    for model_type, special_tokens in SPECIAL_TOKENS.items():
        for token in special_tokens.values():
            if token in completion:
                return model_type
    raise ValueError("Model type not recognized in the completion text.")


def _post_process_bbox(bbox: list[float], images: list[str], model_type: str) -> list[float]:
    """Post-process the bounding box coordinates based on the model type.

    Args:
        bbox (list[float]): The bounding box coordinates.
        model_type (str): The model type, either "internvl3" or "qwen2.5-vl".

    Returns:
        list[float]: The post-processed bounding box coordinates.
    """
    if model_type == "qwen2.5-vl":
        return bbox
    elif model_type == "internvl3":
        # since intervl3 uses `norm1000`, we need to de-normalize the bounding box
        if len(images) != 1:
            raise ValueError("For internvl3, only one image is expected.")
        width, height = imagesize.get(images[0])
        x1, y1, x2, y2 = bbox
        x1, y1 = x1 * width / 1000, y1 * height / 1000
        x2, y2 = x2 * width / 1000, y2 * height / 1000
        return [x1, y1, x2, y2]
    else:
        raise ValueError(f"Model type {model_type} is not supported for post-processing bounding boxes.")


def _cal_bbox_distance(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate the distance between two bounding boxes.

    Args:
        bbox1 (list[float]): The first bounding box coordinates.
        bbox2 (list[float]): The second bounding box coordinates.

    Returns:
        float: The Euclidean distance between the two bounding boxes.
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    center2 = ((x3 + x4) / 2, (y3 + y4) / 2)
    return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5


def _cal_bbox_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bbox1 (list[float]): The first bounding box coordinates.
        bbox2 (list[float]): The second bounding box coordinates.

    Returns:
        float: The IoU value.
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    # Calculate the intersection area
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    inter_left, inter_right = max(x1, x3), min(x2, x4)
    inter_top, inter_bottom = max(y1, y3), min(y2, y4)
    inter_area = max(0, (inter_right - inter_left) * (inter_bottom - inter_top))
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def cal_iou(completion: str, objects: dict, images: list[str]) -> float:
    """Calculate the IoU (Intersection over Union) between the completion and the objects.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".
        images (list[str]): List of image paths corresponding to the completion.

    Returns:
        float: The IoU value.
    """
    # since objects is empty, there is no need to perform grounding
    if not objects:
        return 1.0
    # extract bounding boxes from the completion
    model_type = _get_model_type(completion)
    ref_object_start = SPECIAL_TOKENS[model_type]["ref_object_start"]
    ref_object_end = SPECIAL_TOKENS[model_type]["ref_object_end"]
    box_start = SPECIAL_TOKENS[model_type]["box_start"]
    box_end = SPECIAL_TOKENS[model_type]["box_end"]
    pattern = rf"{ref_object_start}.*?{ref_object_end}\s*{box_start}(.*?){box_end}"
    bboxes = []
    for box in re.findall(pattern, completion):
        # qwen2.5-vl: [[x1, y1], [x2, y2]]
        # internvl3: (x1,y1),(x2,y2)
        box = box.strip().replace(" ", "").replace("[[", "[").replace("]]", "]")
        box = box.replace("(", "[").replace(")", "]")
        # converted to [x1, y1], [x2, y2]
        try:
            x1, y1, x2, y2 = json.loads(f"[{box}]")[0]
            bbox = _post_process_bbox([x1, y1, x2, y2], images, model_type)
            bboxes.append(bbox)
        except Exception:
            continue
    # calculate IoU
    # match the pred and ground truth bboxes using closest distance
    gt_bboxes = objects["bbox"]
    matched_bbox_indices = []
    for i, bbox in enumerate(bboxes):
        min_distance, closest_bbox_index = math.inf, -1
        for j, gt_bbox in enumerate(gt_bboxes):
            if j in matched_bbox_indices:
                continue
            distance = _cal_bbox_distance(bbox, gt_bbox)
            if distance < min_distance:
                min_distance = distance
                closest_bbox_index = j
        matched_bbox_indices.append(closest_bbox_index)
    # calculate IoU for matched bboxes
    iou_sum = 0.0
    for i, j in enumerate(matched_bbox_indices):
        if j == -1:
            continue
        iou_sum += _cal_bbox_iou(bboxes[i], gt_bboxes[j])
    iou = iou_sum / len(gt_bboxes) if gt_bboxes else 1.0
    return iou


def check_format(completion: str, objects: dict) -> bool:
    """Check if the completion format is correct.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".

    Returns:
        bool: True if the format is correct, False otherwise.
    """
    # since objects is empty, the grounding format is not required
    if not objects:
        pattern = r"<think>.*?</think>\s*<verdict>.*?</verdict>"
    else:
        model_type = _get_model_type(completion)
        ref_object_start = SPECIAL_TOKENS[model_type]["ref_object_start"]
        ref_object_end = SPECIAL_TOKENS[model_type]["ref_object_end"]
        box_start = SPECIAL_TOKENS[model_type]["box_start"]
        box_end = SPECIAL_TOKENS[model_type]["box_end"]
        # prepare pattern to match the completion format
        grounding_pattern = f"{ref_object_start}.*?{ref_object_end}\s*{box_start}.*?{box_end}.*?"
        pattern = rf"<think>({grounding_pattern})+.*?<tag>.*?</tag>.*?</think>\s*<verdict>.*?</verdict>"
    # check if the completion matches the pattern
    matches = re.search(pattern, completion, re.DOTALL)
    return matches is not None


def check_label(completion: str, objects: dict) -> bool:
    """Check if the completion contains the correct label.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".

    Returns:
        bool: True if the label is correct, False otherwise.
    """
    # since objects is empty, the ground truth is `real`
    if not objects:
        ground_truth = "real"
    else:
        ground_truth = "generated"
    # extract the label from the completion
    label_match = re.search(r"<verdict>(.*?)</verdict>", completion)
    predicted_label = label_match.group(1) if label_match else ""
    predicted_label = predicted_label.strip().lower()
    # check if the predicted label matches the ground truth
    return predicted_label == ground_truth
