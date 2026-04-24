import re
from app.services.ocr_service import perform_ocr
from app.services.phone_detector import get_phone_bounding_boxes
from app.services.mask_service import create_mask
from app.services.vertex_service import edit_image_with_vertex
from app.utils.image_utils import get_image_dimensions, cv2_to_bytes

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
    # 1. Initial OCR & Detection
    ocr_result = perform_ocr(image_bytes)
    bounding_boxes = get_phone_bounding_boxes(ocr_result)

    if not bounding_boxes:
        raise ValueError("No phone numbers detected in the original image.")

    original_phone_numbers = [box["matched_text"] for box in bounding_boxes]
    width, height = get_image_dimensions(image_bytes)

    for retry_count in range(max_retries):
        try:
            # 2. Mask Generation (increase padding slightly on retries)
            padding = 10 + (retry_count * 5)
            mask_np = create_mask(width, height, bounding_boxes, padding=padding)
            mask_bytes = cv2_to_bytes(mask_np)

            # 3. Image Editing
            edited_image_bytes = edit_image_with_vertex(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                new_phone_number=new_phone_number,
                retry_count=retry_count
            )

            # 4. Validation
            is_valid = validate_edit(edited_image_bytes, original_phone_numbers, new_phone_number)

            if is_valid:
                return edited_image_bytes

            print(f"Validation failed on retry {retry_count}. Retrying...")

        except Exception as e:
            print(f"Error during processing on retry {retry_count}: {e}")

    raise Exception("Failed to successfully replace phone numbers after maximum retries.")
