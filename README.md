# Industrial Defect Detection & Quality Control System

A high-performance automated vision inspection application powered by **FastAPI** and **YOLOv8**, designed for real-time surface defect detection, classification, and cloud deployment.

![Defect Detection](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🌟 Key Features

- **Multi-Modal Detection**:
  - **Surface Classifier (`defect_best.pt`)**: Classifies surface states into `crack`, `hole`, `normal`, `rust`, and `scratch`.
  - **Object Detector (`best.pt`)**: Detects bounding boxes for `dent`, `hole`, `normal`, `rust`, and `scratch`.
- **Interactive Cloud Inspection Dashboard**:
  - **Browser Webcam Tab**: Stream frames directly from any client device camera using HTML5 `getUserMedia()`.
  - **Image Upload Tab**: Drag-and-drop defect test pictures with visual annotated overlays and confidence scores.
  - **Hardware / Server Stream**: Live MJPEG video feed (`/video_feed`) for factory IP / USB webcams.
- **Dynamic Model Switching**: Switch between classification and detection models on-the-fly via UI dropdown or REST API.
- **Cloud Ready**: Configured for 1-click deployment on **Render**, **Railway**, **Hugging Face Spaces**, or any Docker host.

---

## 🚀 Quick Start (Local Run)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```
Or via Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **[http://localhost:8000](http://localhost:8000)**.
Interactive API documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t defect-detector .

# Run container
docker run -p 8000:8000 defect-detector
```

---

## ☁️ Cloud Deployment (Render / Railway / Hugging Face)

See the complete guide in [README_DEPLOYMENT.md](README_DEPLOYMENT.md) for 1-click deployment instructions on:
- **Render** (Free Web Service)
- **Railway**
- **Hugging Face Spaces** (Free Docker Space)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web inspection dashboard (Webcam, Upload, Server Stream) |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/health` | Cloud container health check probe |
| `POST` | `/predict/image` | Multipart image upload for defect detection |
| `POST` | `/predict/frame` | Base64 video frame stream from client webcam |
| `GET` | `/models` | List available models |
| `POST` | `/models/select` | Switch between classification & bounding box detection |
| `GET` | `/detect_status` | JSON telemetry of active detection stream |
| `GET` | `/video_feed` | MJPEG video stream (hardware or cloud standby) |
