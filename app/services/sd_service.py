import requests
import os

def call_sd_api(image_path: str, mask_path: str, new_number: str) -> str:
    """
    Call the external Stable Diffusion Inpainting API to edit the image.

    Args:
        image_path: Path to the input image file.
        mask_path: Path to the binary mask image file.
        new_number: The replacement phone number.

    Returns:
        The path to the saved edited image output.
    """
    api_url = "https://9a8a-136-110-5-22.ngrok-free.app/inpaint"
    output_path = os.path.join("temp", "output.png")

    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

    formatted_prompt = "Remove the text inside the masked region completely.\nFill it with a matching background.\nDo not add any new text.\nDo not modify anything outside the masked region."

    print(f"[SD Service] Calling Stable Diffusion Inpainting API...")
    print(f"[SD Service] Prompt: {formatted_prompt}")

    try:
        with open(image_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
            files = {
                "image": img_file,
                "mask": mask_file
            }
            data = {
                "prompt": formatted_prompt
            }

            # Set a 20 second timeout for the external API call
            response = requests.post(api_url, files=files, data=data, timeout=20)

        # Check for non-200 responses
        response.raise_for_status()

        # Check for empty response
        if not response.content:
            raise ValueError("API returned an empty response.")

        # Save the resulting image
        with open(output_path, "wb") as out_file:
            out_file.write(response.content)

        print(f"[SD Service] Successfully saved SD output to {output_path}")
        return output_path

    except requests.exceptions.Timeout:
        print("[SD Service] Error: Request to Stable Diffusion API timed out.")
        raise Exception("Stable Diffusion API request timed out.")
    except requests.exceptions.ConnectionError as ce:
        print(f"[SD Service] Error: Failed to connect to Stable Diffusion API: {ce}")
        raise Exception(f"Connection error to Stable Diffusion API: {ce}")
    except requests.exceptions.RequestException as re:
        print(f"[SD Service] Error: Stable Diffusion API request failed: {re}")
        raise Exception(f"Stable Diffusion API error: {re}")
    except Exception as e:
        print(f"[SD Service] Unexpected error during SD API call: {e}")
        raise Exception(f"Unexpected error calling Stable Diffusion: {e}")
