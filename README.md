# Object Detection Service (Cascade Person -> PPE)

This service implements a **two-stage cascade detection system**:
1.  **Stage 1:** Detects **People** in the full frame using a standard pre-trained YOLO model from https://github.com/J3lly-Been/YOLOv8-HumanDetection.
2.  **Stage 2:** Crops each person and detects **PPE** (Personal Protective Equipment) specifically on them using a custom trained model.

## Tech Stack

- **Language:** Python 3.11.9
- **Framework:** FastAPI
- **ML Models:** YOLOv8 (Ultralytics) x 2
- **Container:** Docker
- **CI/CD:** GitHub Actions

---

## Quick Start

Prerequisite: Docker must be installed.

```bash
docker run -p 8000:8000 notcoolkid/detector-service:latest
```

- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Model Configuration

The service uses **two** models, configured via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PERSON` | `models/human_detector.pt` | Specialized Human Detection Model. |
| `MODEL_PPE` | `models/ppe_detector.pt` | Our trained PPE-detection model. |

### How to Update Models

1.  **Place Files:**
    - Put human detector at `models/human_detector.pt`.
    - Put PPE detector at `models/ppe_detector.pt`.
2.  **Commit & Push:**
    ```bash
    git add models/human_detector.pt models/ppe_detector.pt
    git commit -m "chore: update detection models"
    git push origin main
    ```
3.  **Deploy:** Manually trigger the GitHub Action "Build and Push Docker Image".

---

## API Response Structure

The API now returns detections with a `type` field ("primary" for person, "secondary" for PPE) and links PPE items to their parent person.

```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.98,
      "bbox": [100, 100, 300, 500],
      "type": "primary"
    },
    {
      "class": "helmet",
      "confidence": 0.95,
      "bbox": [120, 110, 200, 180],
      "type": "secondary",
      "parent_person_bbox": [100, 100, 300, 500]
    }
  ]
}
```

---

## Development Setup

### Option 1: Docker

```bash
docker build -t detector-service .
docker run -p 8000:8000 detector-service
```

### Option 2: Local Python Environment

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Environment:**
    Create a `.env` file in the project root:
    ```env
    MODEL_PERSON=models/human_detector.pt
    MODEL_PPE=models/ppe_detector.pt
    ```
3.  **Run the Server:**
    ```bash
    python main.py
    ```
