import cv2
import numpy as np

from mllm_defake.utils import decode_base64_to_image, encode_image_to_base64


def find_roi_adaptive(image: np.ndarray) -> np.ndarray:
    """
    Find ROI using adaptive thresholding and morphological operations.

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Cropped ROI
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # Block size
        2,  # C constant
    )

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Get largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )

    # Apply perspective transform
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    roi = cv2.warpPerspective(image, m, (width, height))

    return roi


def find_roi_pyramid(image, scale_factor=1.5, min_size=(30, 30)):
    """
    Find ROI using multi-scale processing.

    Args:
        image (np.ndarray): Input image
        scale_factor (float): Factor to reduce image by at each scale
        min_size (tuple): Minimum size of the image to process

    Returns:
        np.ndarray: Cropped ROI
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create image pyramid
    pyramid = []
    current_img = gray.copy()

    while True:
        h, w = current_img.shape[:2]
        if h < min_size[1] or w < min_size[0]:
            break

        pyramid.append(current_img)
        current_img = cv2.resize(current_img, (int(w / scale_factor), int(h / scale_factor)))

    # Process each scale
    roi_candidates = []

    for scaled in pyramid:
        # Apply edge detection
        edges = cv2.Canny(scaled, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour
            largest = max(contours, key=cv2.contourArea)

            # Calculate bounding box
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Scale back to original size
            scale = image.shape[1] / scaled.shape[1]
            scaled_box = box * scale

            roi_candidates.append(scaled_box)

    if not roi_candidates:
        return image

    # Select best candidate (here using the largest area)
    best_roi = max(roi_candidates, key=cv2.contourArea)

    # Extract ROI using perspective transform
    rect = cv2.minAreaRect(best_roi)
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = best_roi.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    roi = cv2.warpPerspective(image, M, (width, height))

    return roi


def adaptive(cache: dict) -> None:
    image = cache["image_url"]
    image = decode_base64_to_image(image)
    roi = find_roi_adaptive(image)
    roi_encoded = encode_image_to_base64(roi)
    cache["roi_url"] = roi_encoded


def pyramid(cache: dict) -> None:
    image = cache["image_url"]
    image = decode_base64_to_image(image)
    roi = find_roi_pyramid(image)
    roi_encoded = encode_image_to_base64(roi)
    cache["roi_url"] = roi_encoded


def dinov2(cache: dict) -> None:
    pass
