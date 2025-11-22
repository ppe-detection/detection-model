## ⚙️ CI/CD Pipeline

The project uses **GitHub Actions** (`.github/workflows/docker-publish.yml`) to automate the build process.

### Workflow
The workflow is set to **Manual Trigger** to prevent accidental deployments.

To release a new version:
1.  Go to the **Actions** tab in GitHub.
2.  Select "Build and Push Docker Image".
3.  Click **Run workflow**.
4.  This will build the image and push it to Docker Hub at `notcoolkid/detector-service:latest`.

### Secrets Configuration
If you fork this repo, you must configure these Repository Secrets in GitHub:
- `DOCKER_USERNAME`: Your Docker Hub username.
- `DOCKER_PASSWORD`: Your Docker Hub Access Token.
