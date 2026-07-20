import io
import os
import uvicorn
import numpy as np
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2

app = FastAPI()

# Create a clean folder framework for processed uploads
os.makedirs("images_uploaded", exist_ok=True)

# Load OpenCV's built-in, lightweight face detector (takes 0MB of RAM)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class ProcessingMode(str, Enum):
    face_detect = "face_detect"
    add_border = "add_border"


@app.get("/")
def home():
    return "Congratulations! Your lightweight API is working perfectly. Head over to /docs"


@app.post("/predict")
def prediction(mode: ProcessingMode, file: UploadFile = File(...)):
    print(f"[DEBUG] Processing file: {file.filename} with mode: {mode}", flush=True)

    filename = file.filename
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    try:
        # 1. Read the uploaded file bytes
        file_bytes = file.file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 2. Decode the bytes into an image matrix
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image format.")

        # 3. Process the image based on user choice (No heavy AI models!)
        if mode == ProcessingMode.face_detect:
            print("[DEBUG] Running lightweight face detection...", flush=True)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            print(f"[DEBUG] Found {len(faces)} face(s). Drawing bounding boxes.", flush=True)
            for (x, y, w, h) in faces:
                # Draw a bright green rectangle around detected faces
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(image, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif mode == ProcessingMode.add_border:
            print("[DEBUG] Adding a stylized frame border...", flush=True)
            # Draw a thick blue border frame inside the image boundaries
            h, w, _ = image.shape
            cv2.rectangle(image, (0, 0), (w, h), (255, 0, 0), 20)
            cv2.putText(image, "PROCESSED BY RENDER", (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # 4. Save the modified output image securely
        output_path = f'images_uploaded/{filename}'
        cv2.imwrite(output_path, image)

        # 5. Stream the modified image directly back to the client
        file_image = open(output_path, mode="rb")
        return StreamingResponse(file_image, media_type="image/jpeg")

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"[ERROR] Pipeline crashed: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
