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

# --- Model Configuration ---
# Define paths and class mappings for each detector
# "classes": { model_class_id: "Output Name" }
# If "classes" is None, all detected classes are returned with their original names.
DETECTOR_CONFIG = {
    "person": {
        "path": os.getenv("MODEL_PERSON", "models/Person_Test.pt"),
        "classes": {0: "Person"}  # Only detect class 0, rename to "Person"
    },
    "eye": {
        "path": os.getenv("MODEL_EYE", "models/Eye_Test.pt"),
        "classes": {0: "Eyes"} # Example renaming
    },
    "glove": {
        "path": os.getenv("MODEL_GLOVE", "models/Glove_Test.pt"),
        "classes": {0: "Glove"}
    },
    "goggles": {
        "path": os.getenv("MODEL_GOGGLES", "models/Goggles_Test.pt"),
        "classes": {0: "Goggles"}
    },
    "hand": {
        "path": os.getenv("MODEL_HAND", "models/Hand_Test.pt"),
        "classes": {0: "Hand"}
    },
    "lab_coat": {
        "path": os.getenv("MODEL_LAB_COAT", "models/Lab_Coat_Test.pt"),
        "classes": {0: "Lab_Coat"}
    }
}

models = {}

def load_models():
    for key, config in DETECTOR_CONFIG.items():
        path = config["path"]
        try:
            print(f"Loading {key} Model: {path}")
            if path.startswith("models/") and not os.path.exists(path):
                print(f"Custom model {path} not found. Falling back to yolov8n.pt")
                models[key] = YOLO("yolov8n.pt")
            else:
                models[key] = YOLO(path)
        except Exception as e:
            print(f"Error loading {key} Model: {e}")
            models[key] = None

# Load all models
load_models()


@app.get("/")
async def root():
    return {
        "message": "Cascade Object Detection Service is running",
        "config": DETECTOR_CONFIG
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if the primary person detector is available
    if not models.get("person"):
        raise HTTPException(status_code=500, detail="Person model not initialized correctly")

    try:
        # 1. Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # Convert to RGB if needed (e.g., for PNGs with alpha channel)
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_width, original_height = image.size

        # 2. Stage 1: Detect People
        person_model = models["person"]
        person_config = DETECTOR_CONFIG["person"]
        
        # Run inference
        person_results = person_model(image, verbose=False)
        
        final_detections = []
        
        # 3. Stage 2: Process each person
        for result in person_results:
            for box in result.boxes:
                # Get person bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf)
                cls_id = int(box.cls)
                
                # Filter/Map Class Name
                if person_config.get("classes"):
                    if cls_id not in person_config["classes"]:
                        continue # Skip irrelevant classes
                    cls_name = person_config["classes"][cls_id]
                else:
                    cls_name = person_model.names[cls_id]
                
                # Add the person detection itself to results
                final_detections.append({
                    "class": cls_name,
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
                
                # Run Secondary detections on the crop
                secondary_models_keys = ["eye", "glove", "goggles", "hand", "lab_coat"]
                
                for model_key in secondary_models_keys:
                    model = models.get(model_key)
                    if not model:
                        continue
                    
                    sub_config = DETECTOR_CONFIG.get(model_key, {})

                    # Run inference
                    sub_results = model(person_crop, verbose=False)
                    
                    # Process sub-detections
                    for sub_res in sub_results:
                        for sub_box in sub_res.boxes:
                            px1, py1, px2, py2 = sub_box.xyxy[0].tolist()
                            sub_conf = float(sub_box.conf)
                            sub_cls_id = int(sub_box.cls)
                            
                            # Filter/Map Sub-Class Name
                            if sub_config.get("classes"):
                                if sub_cls_id not in sub_config["classes"]:
                                    continue
                                sub_cls_name = sub_config["classes"][sub_cls_id]
                            else:
                                sub_cls_name = model.names[sub_cls_id]
                            
                            # MAP COORDINATES BACK TO ORIGINAL IMAGE
                            # The crop coordinates are relative to the crop (0,0 is top-left of person)
                            # We must add the offset (left, top) of the person box
                            final_x1 = px1 + left
                            final_y1 = py1 + top
                            final_x2 = px2 + left
                            final_y2 = py2 + top
                            
                            final_detections.append({
                                "class": sub_cls_name,
                                "confidence": sub_conf,
                                "bbox": [final_x1, final_y1, final_x2, final_y2],
                                "parent_person_bbox": [x1, y1, x2, y2], # Link to the person
                                "type": "secondary",
                                "detector": model_key # Track which detector found this
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
