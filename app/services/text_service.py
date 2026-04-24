from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def draw_text_on_image(img: Image.Image, bbox: dict, new_number: str) -> Image.Image:
    """
    Draws the new phone number onto the PIL image at the specified bounding box.
    Uses PIL to draw text exactly at (x1, y1) scaled to bbox height.
    """
    draw = ImageDraw.Draw(img)

    x1 = int(bbox["min_x"])
    y1 = int(bbox["min_y"])
    x2 = int(bbox["max_x"])
    y2 = int(bbox["max_y"])

    # Calculate font size based on bounding box height
    bbox_height = max(10, y2 - y1)
    font_size = int(bbox_height * 0.8)

    # Load a default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback if arial is not found on the system
        font = ImageFont.load_default()
        # Scale default font manually if needed or just accept the limitation
        # (ImageFont.load_default doesn't support dynamic sizes easily, so we usually rely on system fonts)
        # For production Linux environments, DejavuSans or FreeSans are common
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            pass

    # Basic color detection
    # Convert image to numpy array to sample pixel colors
    img_np = np.array(img)

    # We want text color to contrast with background.
    # Sample a few pixels around the bounding box to guess the background color.
    # Just pick top-left corner just outside bbox as background guess
    bg_y = max(0, y1 - 2)
    bg_x = max(0, x1 - 2)
    bg_color = img_np[bg_y, bg_x]

    # Determine luminance of background to choose text color
    # luminance = 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]

    # Default to black text if background is bright, white if background is dark
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    # Draw text at top-left of bbox
    draw.text((x1, y1), new_number, fill=text_color, font=font)

    return img
