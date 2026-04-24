import re
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from app.services.ocr_service import perform_ocr
from app.services.phone_detector import get_phone_bounding_boxes
from app.services.mask_service import create_mask
from app.services.text_service import draw_text_on_image
from app.utils.image_utils import get_image_dimensions, cv2_to_bytes, bytes_to_cv2

def validate_edit(edited_image_bytes: bytes, original_phone_numbers: list, new_phone_number: str) -> bool:
    """
    Validate that the old numbers are gone and the new number is present.
    """
    ocr_result = perform_ocr(edited_image_bytes)
    edited_text = ocr_result["text"]

    # Strip spaces for easier matching
    clean_edited_text = re.sub(r'\s+', '', edited_text)
    clean_new_phone = re.sub(r'\s+', '', new_phone_number)

    # 1. Ensure new phone number exists in the edited text
    if clean_new_phone not in clean_edited_text:
        return False

    # 2. Ensure old phone numbers are removed
    for old_phone in original_phone_numbers:
        clean_old_phone = re.sub(r'\s+', '', old_phone)
        if clean_old_phone in clean_edited_text:
            return False

    return True

def process_and_validate(image_bytes: bytes, new_phone_number: str, max_retries: int = 3) -> bytes:
    """
    Main pipeline loop with retries.
    """
    print(f"[Validator] Starting process_and_validate. Target replacement number: {new_phone_number}")
    # 1. Initial OCR & Detection
    print("[Validator] Step 1: Running initial OCR to detect phone numbers...")
    ocr_result = perform_ocr(image_bytes)
    bounding_boxes = get_phone_bounding_boxes(ocr_result)

    if not bounding_boxes:
        print("[Validator] No phone numbers detected in the original image.")
        raise ValueError("No phone numbers detected in the original image.")

    original_phone_numbers = [box["matched_text"] for box in bounding_boxes]
    print(f"[Validator] Detected phone numbers: {original_phone_numbers}")

    width, height = get_image_dimensions(image_bytes)

    # Load original image array for inpainting (no resizing)
    base_img_np = bytes_to_cv2(image_bytes)

    padding_steps = [0, 5, 10]

    for retry_count in range(max_retries):
        print(f"\n[Validator] --- Attempt {retry_count + 1} of {max_retries} ---")
        try:
            # 2. Mask Generation (increase padding slightly on retries)
            padding = padding_steps[retry_count] if retry_count < len(padding_steps) else 10
            print(f"[Validator] Step 2: Generating mask with padding = {padding}px")
            mask_np = create_mask(width, height, bounding_boxes, padding=padding)

            # 3. Image Editing via OpenCV Inpainting
            print("[Validator] Step 3: Inpainting image using cv2.INPAINT_TELEA...")
            inpainted_img = cv2.inpaint(base_img_np, mask_np, 3, cv2.INPAINT_TELEA)

            # Convert to PIL for drawing
            inpainted_img_rgb = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(inpainted_img_rgb)

            # Draw new phone number(s) onto the inpainted image
            print("[Validator] Step 3.5: Drawing new text onto inpainted image...")
            for box in bounding_boxes:
                pil_image = draw_text_on_image(pil_image, box, new_phone_number)

            # Convert the final PIL image back into bytes for validation and output
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            edited_image_bytes = img_byte_arr.getvalue()

            # 4. Validation
            print("[Validator] Step 4: Validating edited image via OCR...")
            is_valid = validate_edit(edited_image_bytes, original_phone_numbers, new_phone_number)

            if is_valid:
                print("[Validator] Validation SUCCESS. Returning processed image.")
                return edited_image_bytes

            print(f"[Validator] Validation FAILED on attempt {retry_count + 1}. OCR output did not match expectations.")

        except Exception as e:
            print(f"[Validator] Error during attempt {retry_count + 1}: {e}")

    print("[Validator] Failed to successfully replace phone numbers after maximum retries.")
    raise Exception("Failed to successfully replace phone numbers after maximum retries.")
