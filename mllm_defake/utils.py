import base64
import json
from pathlib import Path

import cv2
import filetype
import numpy as np


def read(p: Path | str, bytes: bool = False) -> str | bytes:
    """
    Read the contents of a file.
    """
    return Path(p).read_text() if not bytes else Path(p).read_bytes()


def pprint_messages_payload(msgs: list) -> str:
    """
    Pretty-print the messages payload.
    """
    ret = "\n=== BEGINNING OF CONVERSATION ===\n"
    turn_count = 0
    for i, msg in enumerate(msgs):
        ret += f"Message {i + 1}:\n"
        role = msg["role"]
        if role == "system" or role == "developer":
            ret += f"  {role.capitalize() + ' ' * (12 - len(role))}: {json.dumps(msg['content'])}\n"
        else:
            turn_count += 1
            ret += f"  {role.capitalize() + ' ' * (12 - len(role))}: {json.dumps(msg['content']) if len(json.dumps(msg['content'])) < 100 else json.dumps(msg['content'])[:100] + '...'}\n"
    ret += f"=== END OF CONVERSATION (Turns: {turn_count}) ==="
    return ret


def encode_image_to_base64(image_path_or_array: str | Path | np.ndarray) -> str:
    """
    Encodes an image from the given file path to base64 format.

    Args:
        image_path: The path to the image file.

    Returns:
        A string in the format "data:image/{image_type};base64,{base64_encoded_image}"
    """
    if isinstance(image_path_or_array, np.ndarray):
        image_array = image_path_or_array
        _, image_data = cv2.imencode(".png", image_array)
        base64_encoded_image = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/png;base64,{base64_encoded_image}"
    image_path = Path(image_path_or_array)
    if not image_path.is_file():
        raise FileNotFoundError(f"File not found: {image_path}")
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_type: filetype.Type = filetype.guess(image_data)
        if image_type:
            base64_encoded_image = base64.b64encode(image_data).decode("utf-8")
            return f"data:{image_type.mime};base64,{base64_encoded_image}"
        raise ValueError("Unsupported image format")


def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decodes a base64 string to an image.

    Args:
        base64_string: The base64 string to decode.

    Returns:
        A NumPy array representing the image.
    """
    b64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    image_array = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
