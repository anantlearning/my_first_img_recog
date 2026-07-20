import io
import os
import uvicorn
import numpy as np
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2

app = FastAPI()

# Create folder for saving modified files
os.makedirs("images_uploaded", exist_ok=True)


class ProcessingMode(str, Enum):
    blur_image = "blur_image"
    add_border = "add_border"


@app.get("/")
def home():
    return "API is completely active. Go to /docs to upload your image!"


@app.post("/predict")
def prediction(mode: ProcessingMode, file: UploadFile = File(...)):
    print(f"[DEBUG] Received image request for mode: {mode}", flush=True)

    filename = file.filename
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=415, detail="Unsupported file format.")

    try:
        # Read the file bytes safely
        file_bytes = file.file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded file has no data.")

        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image layout structure.")

        # Process using bulletproof, basic image-matrix manipulation functions
        if mode == ProcessingMode.blur_image:
            print("[DEBUG] Applying a Gaussian blur matrix filter...", flush=True)
            image = cv2.GaussianBlur(image, (25, 25), 0)
            cv2.putText(image, "BLUR FILTER APPLIED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        elif mode == ProcessingMode.add_border:
            print("[DEBUG] Injecting a frame border blueprint...", flush=True)
            h, w, _ = image.shape
            # Draw a thick blue border around the edge of the matrix frame
            cv2.rectangle(image, (0, 0), (w, h), (255, 0, 0), 25)
            cv2.putText(image, "PROCESSED ON RENDER", (40, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Save the frame to the server drive
        output_path = f'images_uploaded/{filename}'
        cv2.imwrite(output_path, image)

        # Stream back to client
        file_image = open(output_path, mode="rb")
        return StreamingResponse(file_image, media_type="image/jpeg")

    except Exception as e:
        print(f"[ERROR] Process crashed: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")
