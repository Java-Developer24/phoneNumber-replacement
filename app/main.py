from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.services.validator import process_and_validate
from app.config import settings
import uuid
import os

app = FastAPI(title="Phone Number Replacer AI")

# Allow CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

@app.post("/process-image")
async def process_image_endpoint(
    image: UploadFile = File(...),
    replacement_number: str = Form(...)
):
    """
    Endpoint to process an image, replacing all detected phone numbers with the `replacement_number`.
    """
    # Validate file size (e.g. 5MB)
    MAX_SIZE = 5 * 1024 * 1024 # 5 MB

    file_bytes = await image.read()
    if len(file_bytes) > MAX_SIZE:
        return JSONResponse(status_code=400, content={"error": "File size exceeds 5MB limit."})

    # Generate unique filenames for debug/temp
    job_id = str(uuid.uuid4())
    print(f"\n[API] Received new request: Job ID {job_id} | Target Replacement Number: {replacement_number} | File: {image.filename}")

    try:
        # Process and validate
        processed_bytes = process_and_validate(file_bytes, replacement_number)

        # We can optionally save it temporarily, but returning directly is faster.
        print(f"[API] Job ID {job_id} completed successfully. Returning image bytes.")
        return Response(content=processed_bytes, media_type=image.content_type)

    except ValueError as ve:
        # Meaning no phone numbers detected
        print(f"[API] Job ID {job_id} rejected: {ve}")
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        # Logs could be added here for failures
        print(f"[API] Job ID {job_id} failed with exception: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to process image.", "details": str(e)})

# Mount frontend directory
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
