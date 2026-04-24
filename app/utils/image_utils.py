import cv2
import numpy as np

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to an OpenCV image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cv2_to_bytes(img: np.ndarray, ext: str = '.png') -> bytes:
    """Convert OpenCV image to bytes."""
    success, encoded_image = cv2.imencode(ext, img)
    if not success:
        raise ValueError("Could not encode image")
    return encoded_image.tobytes()

def get_image_dimensions(image_bytes: bytes) -> tuple:
    """Return width and height of an image from bytes."""
    img = bytes_to_cv2(image_bytes)
    height, width = img.shape[:2]
    return width, height
