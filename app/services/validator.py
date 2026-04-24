import re
import os
import uuid
from app.services.ocr_service import perform_ocr
from app.services.phone_detector import get_phone_bounding_boxes
from app.services.mask_service import create_mask
from app.services.sd_service import call_sd_api
from app.services.text_service import draw_text_on_image
from app.utils.image_utils import get_image_dimensions, cv2_to_bytes, bytes_to_cv2, resize_to_512

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

    # Base image resizing
    base_img_np = bytes_to_cv2(image_bytes)
    base_img_512 = resize_to_512(base_img_np)
    base_image_bytes_512 = cv2_to_bytes(base_img_512)

    # Calculate resizing scale and offset so we can map bounding boxes accurately for drawing later
    scale = min(512 / width, 512 / height)
    new_w = int(width * scale)
    new_h = int(height * scale)
    x_offset = (512 - new_w) // 2
    y_offset = (512 - new_h) // 2

    padding_steps = [0, 5, 10]

    for retry_count in range(max_retries):
        print(f"\n[Validator] --- Attempt {retry_count + 1} of {max_retries} ---")
        try:
            # 2. Mask Generation (increase padding slightly on retries)
            padding = padding_steps[retry_count] if retry_count < len(padding_steps) else 10
            print(f"[Validator] Step 2: Generating mask with padding = {padding}px")
            mask_np = create_mask(width, height, bounding_boxes, padding=padding)

            # Mask resizing
            mask_np_512 = resize_to_512(mask_np)
            mask_bytes_512 = cv2_to_bytes(mask_np_512)

            # 3. Image Editing
            print("[Validator] Step 3: Dispatching to external Stable Diffusion API for image editing...")

            # Create temp files for the external API call
            temp_id = str(uuid.uuid4())
            temp_image_path = os.path.join("temp", f"input_{temp_id}.png")
            temp_mask_path = os.path.join("temp", f"mask_{temp_id}.png")

            try:
                with open(temp_image_path, "wb") as f:
                    f.write(base_image_bytes_512)
                with open(temp_mask_path, "wb") as f:
                    f.write(mask_bytes_512)

                output_path = call_sd_api(
                    image_path=temp_image_path,
                    mask_path=temp_mask_path,
                    new_number=new_phone_number
                )

                # Draw new phone number(s) onto the SD-cleaned image
                print("[Validator] Step 3.5: Drawing new text onto cleaned image...")
                for box in bounding_boxes:
                    # Map the original bounding box to the 512x512 scaled and offset image
                    scaled_box = {
                        "min_x": (box["min_x"] * scale) + x_offset,
                        "min_y": (box["min_y"] * scale) + y_offset,
                        "max_x": (box["max_x"] * scale) + x_offset,
                        "max_y": (box["max_y"] * scale) + y_offset
                    }
                    draw_text_on_image(output_path, scaled_box, new_phone_number)

                # Read the fully edited and drawn image back into bytes
                with open(output_path, "rb") as f:
                    edited_image_bytes = f.read()

            finally:
                # Cleanup temp files
                for path in [temp_image_path, temp_mask_path, os.path.join("temp", "output.png")]:
                    if os.path.exists(path):
                        os.remove(path)

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
