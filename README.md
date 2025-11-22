# Object Detection Service

This service provides a REST API for object detection using YOLOv8. It is containerized with Docker and includes a fully automated CI/CD pipeline via GitHub Actions.

## Tech Stack

- **Language:** Python 3.11.9
- **Framework:** FastAPI
- **ML Model:** YOLOv8 (Ultralytics)
- **Container:** Docker
- **CI/CD:** GitHub Actions

---

## 🚀 Quick Start (Run the latest version)

Prerequisite: Docker must be installed.

```bash
docker run -p 8000:8000 notcoolkid/detector-service:latest
```

- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/](http://localhost:8000/)

---

## 🧠 Model Management

The service is pre-configured to use a custom trained model located at `models/best.pt`.

### How to Update the Model

1.  **Replace the file:** Overwrite `models/best.pt` with your new trained model file.
2.  **Commit & Push:**
    ```bash
    git add models/best.pt
    git commit -m "chore: update model to v2"
    git push origin main
    ```
3.  **Deploy (Manual Trigger):**
    - Go to the **Actions** tab in this GitHub repository.
    - Select the **Build and Push Docker Image** workflow on the left.
    - Click the **Run workflow** button.
    - This will build the new image with your updated model and push it to Docker Hub.
4.  **Update Production:**
    - Run `docker pull notcoolkid/detector-service:latest` on your server/machine to get the new model.

*(Note: If `models/best.pt` is missing, the service will fallback to `yolov8n.pt` (Nano) to prevent crashing.)*

---

## 🛠️ Development Setup

### Option 1: Docker (Recommended)

```bash
# Build the image locally
docker build -t detector-service .

# Run the container
docker run -p 8000:8000 detector-service
```

### Option 2: Local Python Environment

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Server:**
    ```bash
    python main.py
    ```

---

## ⚙️ CI/CD Pipeline

The project uses **GitHub Actions** (`.github/workflows/docker-publish.yml`) to automate the build process.

### Workflow Strategy
The workflow is set to **Manual Trigger** (`workflow_dispatch`) to prevent accidental deployments during development. It does NOT run automatically on push.

### Secrets Configuration
If you fork this repo, you must configure these Repository Secrets in GitHub:
- `DOCKER_USERNAME`: Your Docker Hub username.
- `DOCKER_PASSWORD`: Your Docker Hub Access Token.

---

## 📡 API Usage

### `POST /predict`

Upload an image file to detect objects.

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

**Example Response:**
```json
{
  "filename": "image.jpg",
  "detections": [
    {
      "class": "helmet",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 250]
    }
  ],
  "count": 1
}
```
