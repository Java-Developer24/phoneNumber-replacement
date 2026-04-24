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

def resize_to_512(img: np.ndarray) -> np.ndarray:
    """Resize an image to exactly 512x512 while maintaining aspect ratio (padding with black)."""
    h, w = img.shape[:2]

    # Calculate scale to fit within 512x512
    scale = min(512 / w, 512 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a 512x512 canvas
    if len(img.shape) == 3:
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((512, 512), dtype=np.uint8)

    # Center the resized image on the canvas
    x_offset = (512 - new_w) // 2
    y_offset = (512 - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas
