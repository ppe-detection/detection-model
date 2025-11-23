from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="YOLO Cascade Object Detection Service")

# --- Model 1: Person Detector ---
MODEL_PERSON_NAME = os.getenv("MODEL_PERSON", "models/human_detector.pt")
try:
    print(f"Loading Person Model: {MODEL_PERSON_NAME}")
    # Fallback for person model
    if MODEL_PERSON_NAME.startswith("models/") and not os.path.exists(MODEL_PERSON_NAME):
        print(f"Custom Person model {MODEL_PERSON_NAME} not found. Falling back to yolov8n.pt")
        MODEL_PERSON_NAME = "yolov8n.pt"
        
    person_model = YOLO(MODEL_PERSON_NAME)
except Exception as e:
    print(f"Error loading Person Model: {e}")
    person_model = None

# --- Model 2: PPE Detector ---
MODEL_PPE_NAME = os.getenv("MODEL_PPE", "models/ppe_detector.pt")
try:
    print(f"Loading PPE Model: {MODEL_PPE_NAME}")
    # Fallback logic for development
    if MODEL_PPE_NAME.startswith("models/") and not os.path.exists(MODEL_PPE_NAME):
         print(f"Custom PPE model {MODEL_PPE_NAME} not found. Falling back to standard yolov8n.pt temporarily.")
         MODEL_PPE_NAME = "yolov8n.pt"
         
    ppe_model = YOLO(MODEL_PPE_NAME)
except Exception as e:
    print(f"Error loading PPE Model: {e}")
    ppe_model = None


@app.get("/")
async def root():
    return {
        "message": "Cascade Object Detection Service is running",
        "models": {
            "person_detector": MODEL_PERSON_NAME,
            "ppe_detector": MODEL_PPE_NAME
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not person_model or not ppe_model:
        raise HTTPException(status_code=500, detail="Models not initialized correctly")

    try:
        # 1. Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # Convert to numpy array for OpenCV cropping logic if needed, 
        # but PIL is fine for YOLO input.
        # We need the original size to handle bounding box mapping if we were resizing,
        # but YOLO handles PIL images natively.
        
        original_width, original_height = image.size

        # 2. Stage 1: Detect People
        # class=0 is typically 'person' in COCO dataset (used by standard YOLOv8)
        person_results = person_model(image, classes=[0], verbose=False)
        
        final_detections = []
        
        # 3. Stage 2: Process each person
        for result in person_results:
            for box in result.boxes:
                # Get person bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf)
                
                # Add the person detection itself to results
                final_detections.append({
                    "class": "person",
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "type": "primary"
                })

                # Crop the person image
                # PIL crop: (left, top, right, bottom)
                # Ensure coordinates are within bounds
                left = max(0, x1)
                top = max(0, y1)
                right = min(original_width, x2)
                bottom = min(original_height, y2)
                
                # Skip if crop is too small
                if (right - left) < 10 or (bottom - top) < 10:
                    continue

                person_crop = image.crop((left, top, right, bottom))
                
                # Run PPE detection on the crop
                ppe_results = ppe_model(person_crop, verbose=False)
                
                # Process PPE detections
                for ppe_res in ppe_results:
                    for ppe_box in ppe_res.boxes:
                        px1, py1, px2, py2 = ppe_box.xyxy[0].tolist()
                        ppe_conf = float(ppe_box.conf)
                        ppe_cls = ppe_model.names[int(ppe_box.cls)]
                        
                        # MAP COORDINATES BACK TO ORIGINAL IMAGE
                        # The crop coordinates are relative to the crop (0,0 is top-left of person)
                        # We must add the offset (left, top) of the person box
                        final_x1 = px1 + left
                        final_y1 = py1 + top
                        final_x2 = px2 + left
                        final_y2 = py2 + top
                        
                        final_detections.append({
                            "class": ppe_cls,
                            "confidence": ppe_conf,
                            "bbox": [final_x1, final_y1, final_x2, final_y2],
                            "parent_person_bbox": [x1, y1, x2, y2], # Link to the person
                            "type": "secondary"
                        })

        return {
            "filename": file.filename,
            "detections": final_detections,
            "count": len(final_detections)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
