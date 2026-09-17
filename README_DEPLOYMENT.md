# Industrial Defect Detector - Cloud Deployment Guide

This project is fully packaged and ready for 1-click cloud deployment on **Render**, **Railway**, **Hugging Face Spaces**, or any **Docker container host**.

---

## 🚀 Option 1: Deploy Free on Render (Recommended)

Render offers a 100% free web service tier suitable for FastAPI and PyTorch CPU inference.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy Defect Detector"
   git branch -M main
   git remote add origin https://github.com/<your-username>/defect-detector.git
   git push -u origin main
   ```
2. **Log into [render.com](https://render.com)**.
3. Click **New +** &rarr; **Web Service**.
4. Connect your GitHub repository.
5. Fill in the service configuration:
   - **Name**: `defect-detector`
   - **Region**: Choose the closest region (e.g. Oregon, Frankfurt, Singapore)
   - **Environment**: `Python` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
6. Click **Create Web Service**.
7. Render will automatically build the environment, install the CPU PyTorch wheels, and launch your service at `https://defect-detector-xxxx.onrender.com`.

---

## 🚂 Option 2: Deploy on Railway

Railway automatically detects either the `Dockerfile` or `Procfile`.

### Steps:
1. Go to [railway.app](https://railway.app) and sign in.
2. Click **New Project** &rarr; **Deploy from GitHub repo**.
3. Select your repository.
4. Railway will automatically pick up the `Dockerfile` or `Procfile`, bind the dynamic `PORT`, and deploy.
5. In the service **Settings**, click **Generate Domain** to get your public HTTPS URL.

---

## 🤗 Option 3: Deploy on Hugging Face Spaces (Free Docker Space)

Hugging Face provides free 2 vCPU / 16 GB RAM Docker spaces with zero cold starts!

### Steps:
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Docker** as the Space SDK (Blank template).
3. Set Space hardware to **Free CPU basic**.
4. Clone the Space repo or push this codebase to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/defect-detector
   git push space main
   ```
5. Hugging Face Spaces will build the `Dockerfile` and expose port `7860` (or `8000`), giving you a permanent free live demo!

---

## 🐳 Option 4: Run Locally with Docker

You can build and test the exact container image on your own machine:

```bash
# Build the Docker container
docker build -t defect-detector .

# Run the container
docker run -p 8000:8000 defect-detector
```
Then open `http://localhost:8000` in your browser.

---

## 📡 REST API Reference

The FastAPI service exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web inspection dashboard (Webcam, Upload, Server Stream) |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/health` | Cloud health check probe (`{"status": "healthy"}`) |
| `POST` | `/predict/image` | Multipart upload of an image file; returns JSON defects & annotated base64 image |
| `POST` | `/predict/frame` | Base64 video frame stream from browser client |
| `GET` | `/models` | List available models (`defect_best.pt`, `best.pt`) |
| `POST` | `/models/select` | Switch between Classification and Bounding Box detection |
| `GET` | `/detect_status` | JSON telemetry of active detection stream |
| `GET` | `/video_feed` | MJPEG video stream (hardware or cloud standby) |
