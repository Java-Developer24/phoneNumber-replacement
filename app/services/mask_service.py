import cv2
import numpy as np

def create_mask(width: int, height: int, bounding_boxes: list, padding: int = 10) -> np.ndarray:
    """
    Create a binary mask image for Vertex AI editing.
    White (255) = replace region
    Black (0) = preserve region
    """
    # Create black image (0s)
    mask = np.zeros((height, width), dtype=np.uint8)

    for box in bounding_boxes:
        # Apply padding, ensuring we don't go out of bounds
        min_x = max(0, box["min_x"] - padding)
        min_y = max(0, box["min_y"] - padding)
        max_x = min(width, box["max_x"] + padding)
        max_y = min(height, box["max_y"] + padding)

        # Fill bounding box with white (255)
        cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, -1)

    return mask
