from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def extract_text_color(original_img_np: np.ndarray, bbox: dict) -> tuple:
    """
    Extracts the dominant text color from the original image inside the bounding box.
    Uses thresholding to separate foreground (text) from background.
    """
    x1 = max(0, int(bbox["min_x"]))
    y1 = max(0, int(bbox["min_y"]))
    x2 = min(original_img_np.shape[1], int(bbox["max_x"]))
    y2 = min(original_img_np.shape[0], int(bbox["max_y"]))

    # Crop the region of interest
    roi = original_img_np[y1:y2, x1:x2]

    if roi.size == 0:
        return (0, 0, 0)

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to separate text from background
    # Usually, text is darker or lighter. Otsu's finds the optimal threshold.
    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # We don't know if text is white-on-black or black-on-white.
    # Assume the smaller area is the text (typical for numbers on a background).
    num_white = cv2.countNonZero(thresh)
    num_black = thresh.size - num_white

    if num_white < num_black:
        # White is the text (foreground mask)
        text_mask = (thresh == 255)
    else:
        # Black is the text (foreground mask)
        text_mask = (thresh == 0)

    # Extract the original color pixels using the text mask
    text_pixels = roi[text_mask]

    if len(text_pixels) == 0:
        return (0, 0, 0)

    # Find the median color of the text pixels to avoid outliers
    median_color = np.median(text_pixels, axis=0)

    # OpenCV uses BGR, PIL uses RGB. Return RGB.
    return (int(median_color[2]), int(median_color[1]), int(median_color[0]))

def draw_text_on_image(img: Image.Image, original_img_np: np.ndarray, bbox: dict, new_number: str) -> Image.Image:
    """
    Draws the new phone number onto the PIL image at the specified bounding box.
    Matches extracted text color, adds pseudo-bold rendering, and a slight shadow for realism.
    """
    # Create an RGBA overlay for drawing text with potential shadow transparency
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    x1 = int(bbox["min_x"])
    y1 = int(bbox["min_y"])
    x2 = int(bbox["max_x"])
    y2 = int(bbox["max_y"])

    # Extract dominant text color
    text_color = extract_text_color(original_img_np, bbox)

    # Calculate font size based on bounding box height
    bbox_height = max(10, y2 - y1)
    font_size = int(bbox_height * 0.85)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

    # Calculate text placement
    text_bbox = draw.textbbox((0, 0), new_number, font=font)
    text_w = text_bbox[2] - text_bbox[0]

    # Center text horizontally in the bounding box
    box_w = max(10, x2 - x1)
    x_pos = x1 + (box_w - text_w) // 2
    # Adjust y to center vertically a bit better
    y_pos = y1 + int(bbox_height * 0.05)

    # 1. Draw shadow
    # Determine shadow color based on text luminance (dark shadow for light text, light glow for dark text)
    luminance = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
    shadow_color = (0, 0, 0, 100) if luminance > 128 else (255, 255, 255, 100)
    shadow_offset = max(1, int(font_size * 0.05))
    draw.text((x_pos + shadow_offset, y_pos + shadow_offset), new_number, fill=shadow_color, font=font)

    # 2. Draw pseudo-bold text (multiple offsets)
    bold_offset = max(1, int(font_size * 0.02))
    offsets = [(0, 0), (bold_offset, 0), (0, bold_offset), (bold_offset, bold_offset)]
    for ox, oy in offsets:
        draw.text((x_pos + ox, y_pos + oy), new_number, fill=text_color + (255,), font=font)

    # Merge overlay with original
    final_img = Image.alpha_composite(img, overlay).convert("RGB")

    return final_img
