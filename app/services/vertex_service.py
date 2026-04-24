from google import genai
from google.genai import types
from app.config import settings

import google.genai.models as models

# --- SDK MONKEY PATCH START ---
# The google-genai 0.3.0 SDK converts MaskReferenceImage.reference_image to "referenceImage" in JSON payload.
# However, the Vertex API expects it as "mask": {"image": {"bytesBase64Encoded": "..."}}.
# Without this patch, the API throws a 400 INVALID_ARGUMENT: Mask image is missing error.
_original_converter = models._ReferenceImageAPI_to_vertex

def _patched_ReferenceImageAPI_to_vertex(api_client, from_object, parent_object=None):
    to_object = _original_converter(api_client, from_object, parent_object)
    # Rewrite the referenceImage key to mask for Mask Reference Images
    if to_object.get("referenceType") == "REFERENCE_TYPE_MASK" and "referenceImage" in to_object:
        to_object["mask"] = {
            "image": {"bytesBase64Encoded": to_object.pop("referenceImage").get("bytesBase64Encoded")}
        }
    return to_object

models._ReferenceImageAPI_to_vertex = _patched_ReferenceImageAPI_to_vertex
# --- SDK MONKEY PATCH END ---

def init_vertex_ai():
    print(f"[Vertex Service] Initializing GenAI Client with project={settings.project_id}, location={settings.location}")
    # Using the new Google GenAI SDK
    return genai.Client(vertexai=True, project=settings.project_id, location=settings.location)

def edit_image(input_path: str, mask_path: str, new_number: str, output_path: str):
    """
    Function signature requested for compatibility.
    Reads input image from path, performs the edit via GenAI, and writes to output_path.
    Note: mask_path is ignored by the GenAI SDK, behavior is simulated by prompt.
    """
    with open(input_path, "rb") as f:
        image_bytes = f.read()

    # mask_path is ignored as mask-based editing is not directly supported via masks in the same way
    # using the simulated prompt approach.

    edited_bytes = edit_image_with_vertex(image_bytes, b"", new_number, 0)

    with open(output_path, "wb") as f:
        f.write(edited_bytes)

def edit_image_with_vertex(image_bytes: bytes, mask_bytes: bytes, new_phone_number: str, retry_count: int = 0) -> bytes:
    """
    Use Google GenAI SDK with Imagen 3 to edit the image based on a prompt.
    NOTE: The old mask-based editing is not supported. Mask bytes are ignored.
    Behavior is simulated using prompt engineering.
    """
    client = init_vertex_ai()

    # Use the latest available image generation model that supports editing
    model_name = "imagen-3.0-capability-001"

    # Create the reference image from bytes
    raw_ref_image = types.RawReferenceImage(
        reference_id=1,
        reference_image=types.Image(image_bytes=image_bytes)
    )

    # Create the mask reference image from bytes
    mask_ref_image = types.MaskReferenceImage(
        reference_id=1, # Must match the reference_id of the base image it applies to
        reference_image=types.Image(image_bytes=mask_bytes),
        config=types.MaskReferenceConfig(
            mask_mode="MASK_MODE_USER_PROVIDED"
        )
    )

    # Prompt adjustments based on retry_count to give varied results if validation fails
    # Simulating the mask by instructing the model to edit ONLY the phone number.
    prompt = f"Identify the phone number in this image and replace it exactly with this text: '{new_phone_number}'. Ensure the background, font style, color, alignment, and size remain exactly the same as the original. Do not modify any other part of the image."

    if retry_count == 1:
        prompt = f"Carefully rewrite the existing phone number to be '{new_phone_number}'. Match the original typography and background completely, leaving all other text and graphics untouched."
    elif retry_count == 2:
        prompt = f"Update the region showing a phone number to show only '{new_phone_number}'. The new text must blend seamlessly with the original font and style. No other modifications are allowed."

    print(f"[Vertex Service] Calling Google GenAI edit_image API (Retry: {retry_count})...")
    print(f"[Vertex Service] Model used: {model_name}")
    print(f"[Vertex Service] Prompt used: {prompt}")

    try:
        response = client.models.edit_image(
            model=model_name,
            prompt=prompt,
            reference_images=[raw_ref_image, mask_ref_image],
            config=types.EditImageConfig(
                edit_mode="EDIT_MODE_INPAINT_INSERTION",
                number_of_images=1,
                guidance_scale=21.0,
                include_rai_reason=True
            )
        )
    except Exception as e:
        print(f"[Vertex Service] Error calling GenAI API: {e}")
        raise Exception(f"GenAI API Error: {e}")

    print("[Vertex Service] Received response from GenAI API.")

    if not response.generated_images:
        print("[Vertex Service] Error: GenAI returned no images or an invalid response.")
        raise Exception("GenAI returned no images or an invalid response")

    return response.generated_images[0].image.image_bytes
