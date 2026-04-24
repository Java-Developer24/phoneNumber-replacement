from vertexai.preview.vision_models import Image, ImageGenerationModel
import vertexai
from app.config import settings

def init_vertex_ai():
    vertexai.init(project=settings.project_id, location=settings.location)

def edit_image_with_vertex(image_bytes: bytes, mask_bytes: bytes, new_phone_number: str, retry_count: int = 0) -> bytes:
    """
    Use Vertex AI Imagen to edit the image based on the mask and prompt.
    """
    init_vertex_ai()

    # We use imagen-edit model
    model = ImageGenerationModel.from_pretrained("imagegeneration@006") # or another appropriate edit model

    base_image = Image(image_bytes)
    mask_image = Image(mask_bytes)

    # Prompt adjustments based on retry_count to give varied results if validation fails
    prompt = f"Replace the text in the masked area exactly with this text: '{new_phone_number}'. Ensure the background, font style, color, alignment, and size remain exactly the same as the original. Do not modify anything outside the mask."

    if retry_count == 1:
        prompt = f"Carefully write the text '{new_phone_number}' in the masked area. Match the original typography and background completely."
    elif retry_count == 2:
        prompt = f"Update the masked region to show only '{new_phone_number}'. The text must blend seamlessly with the original font and style."

    response = model.edit_image(
        base_image=base_image,
        mask=mask_image,
        prompt=prompt,
        # Set parameters for better quality
        number_of_images=1,
        guidance_scale=21, # can be adjusted
    )

    if not response.images:
        raise Exception("Vertex AI returned no images")

    return response.images[0]._image_bytes
