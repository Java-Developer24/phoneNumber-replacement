# Phone Number Replacer AI

This is a production-ready Python backend system that automates replacing phone numbers inside images using Google Cloud Vision API for OCR and Vertex AI Imagen for AI image editing.

## Prerequisites
- Python 3.9+
- A Google Cloud Project with the following APIs enabled:
  - Cloud Vision API
  - Vertex AI API
- A Service Account with appropriate permissions downloaded as a JSON file.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables:**
   Create a `.env` file in the root directory (you can copy `.env.example`).
   You must add the following variables into your `.env` file for the AI services to authenticate properly:
   ```env
   # Path to the downloaded GCP Service Account JSON key
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json

   # Your Google Cloud Project ID (e.g. "my-awesome-project-12345")
   PROJECT_ID=your-google-cloud-project-id

   # The GCP region where you want to run Vertex AI (e.g. "us-central1")
   LOCATION=us-central1
   ```

## Running the Application

Start the FastAPI server using `uvicorn`:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing the Frontend

1. Ensure the server is running.
2. Open your web browser and navigate to `http://localhost:8000`.
3. Use the web interface to upload an image and input a replacement phone number.
4. The frontend will communicate with the backend API and display the original alongside the newly generated processed image!

## API Testing Steps

You can also test the backend application directly using Swagger UI or Postman/cURL.

**Using Swagger UI:**
1. Navigate to `http://localhost:8000/docs` in your browser.
2. Expand the `POST /process-image` endpoint.
3. Click "Try it out".
4. Upload an image containing a phone number.
5. Provide a `replacement_number` (e.g. "1-800-555-0199").
6. Execute and check the returned processed image.

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/process-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/your/image.jpg" \
  -F "replacement_number=1-800-555-0199" --output output.jpg
```

## Notes on Improving Accuracy

To achieve >90% accuracy in real-world scenarios:
- **Mask Generation**: The current padding mechanism is simple but can be improved using morphological operations (dilation) via OpenCV to ensure the mask perfectly envelops text artifacts.
- **OCR Enhancements**: Pre-process noisy images (e.g. binarization, de-skewing) before passing them to the Vision API to improve detection accuracy of split numbers.
- **Model Tuning**: Vertex AI Imagen prompts are currently static (with slight variations on retries). Integrating dynamic prompt generation that assesses the image context (e.g., detecting background color and passing it to the prompt) will yield better blending.
- **Validation Strictness**: The OCR validation currently strips spaces. For exact layout preservation checking, we could check the bounding box dimensions of the newly inserted text against the old text.
