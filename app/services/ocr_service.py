from google.cloud import vision

def perform_ocr(image_bytes: bytes) -> dict:
    """
    Perform DOCUMENT_TEXT_DETECTION using Google Cloud Vision API.
    Returns the full document hierarchy with bounding boxes.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)

    print("[OCR Service] Connecting to GCP Cloud Vision API for DOCUMENT_TEXT_DETECTION...")
    # Use document text detection for structured hierarchy
    response = client.document_text_detection(image=image)
    print("[OCR Service] Received response from Cloud Vision API.")

    if response.error.message:
        print(f"[OCR Service] Vision API Error: {response.error.message}")
        raise Exception(f"Vision API Error: {response.error.message}")

    document = response.full_text_annotation

    # We will format this into an easier structure: a list of words with their text and bounding box.
    # A "word" in Vision API has a bounding box.
    words_info = []

    if not document:
        return {"text": "", "words": words_info}

    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    # Bounding box points
                    vertices = word.bounding_box.vertices
                    box = [
                        {"x": v.x, "y": v.y} for v in vertices
                    ]

                    # Compute min/max to simplify bounding box
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]

                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)

                    words_info.append({
                        "text": word_text,
                        "box": box,
                        "min_x": min_x,
                        "max_x": max_x,
                        "min_y": min_y,
                        "max_y": max_y
                    })

    return {
        "text": document.text,
        "words": words_info
    }
