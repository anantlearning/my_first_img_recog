import io
import os
import uvicorn
import numpy as np
import nest_asyncio
import subprocess
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# 1. ADD THIS CRITICAL IMPORT RIGHT HERE
import cv2

# 2. Keep the cvlib NumPy compatibility patch
if not hasattr(np, 'int'):
    np.int = int

import cvlib as cv
from cvlib.object_detection import draw_bbox


# Assign an instance of the FastAPI class to the variable "app".
# You will interact with your api using this instance.
app = FastAPI(title='Deploying an ML Model with FastAPI')

# Ensure the upload directory exists so Render does not crash
os.makedirs("images_uploaded", exist_ok=True)


# List available models using Enum for convenience. This is useful when the options are pre-defined.
class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://serve/docs"


# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):
    print(f"[DEBUG] Received file: {file.filename} using model: {model}", flush=True)

    filename = file.filename
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    try:
        # 1. READ ALL BYTES SECURELY
        file_bytes = file.file.read()
        if not file_bytes:
            print("[ERROR] File payload came back completely empty.", flush=True)
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 2. DECODE BYTES INTO OPEN_CV IMAGE MATRIX
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. SAFETY GUARD: If image matrix fails to decode, stop here before cvlib runs
        if image is None or image.size == 0:
            print("[ERROR] OpenCV could not read or decode image pixels.", flush=True)
            raise HTTPException(status_code=400, detail="Invalid image formatting or corrupted file.")

        print(f"[DEBUG] Image verified. Shape matches: {image.shape}", flush=True)

        # 4. RUN OBJECT DETECTION SAFELY
        print(f"[DEBUG] Passing verified matrix to cv.detect_common_objects...", flush=True)
        bbox, label, conf = cv.detect_common_objects(image, model=model)
        print(f"[DEBUG] Detection complete! Objects found: {label}", flush=True)

        # 5. DRAW BOUNDARIES AND PROCESS RESPONSE
        output_image = draw_bbox(image, bbox, label, conf)
        output_path = f'images_uploaded/{filename}'
        cv2.imwrite(output_path, output_image)

        # Open and return the modified image binary back to your browser/client
        file_image = open(output_path, mode="rb")
        return StreamingResponse(file_image, media_type="image/jpeg")

    except HTTPException as http_ex:
        # Pass user validation errors straight back out
        raise http_ex
    except Exception as e:
        print(f"[CRITICAL FAILURE] Pipeline crashed on line: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error on Processing: {str(e)}")

