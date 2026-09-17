import base64
import os
import threading
import time
from typing import Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Real-time vision defect inspection powered by YOLOv8 and FastAPI.",
    version="2.0.0"
)

# Enable CORS for cloud & client browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available models
AVAILABLE_MODELS = {
    "defect_best": {
        "filename": "defect_best.pt",
        "name": "Defect Classifier (YOLOv8-cls)",
        "type": "classify",
        "description": "Surface classification: crack, hole, normal, rust, scratch",
    },
    "best": {
        "filename": "best.pt",
        "name": "Defect Bounding Box Detector (YOLOv8-det)",
        "type": "detect",
        "description": "Object detection with bounding boxes: dent, hole, normal, rust, scratch",
    },
}

current_model_id = "defect_best" if os.path.exists(os.path.join(BASE_DIR, "defect_best.pt")) else "best"
model_lock = threading.Lock()

# Load initial model
def load_yolo_model(model_id: str):
    config = AVAILABLE_MODELS.get(model_id)
    if not config:
        raise ValueError(f"Unknown model: {model_id}")
    path = os.path.join(BASE_DIR, config["filename"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return YOLO(path)

try:
    active_yolo = load_yolo_model(current_model_id)
except Exception as e:
    print(f"[WARN] Failed to load model {current_model_id}: {e}")
    active_yolo = None

# Hardware camera detection (Graceful fallback for cloud servers without physical webcams)
camera = None
has_hardware_cam = False

try:
    test_cap = cv2.VideoCapture(0)
    if test_cap.isOpened():
        ret, _ = test_cap.read()
        if ret:
            camera = test_cap
            has_hardware_cam = True
            print("[INFO] Hardware webcam detected and initialized.")
        else:
            test_cap.release()
    else:
        test_cap.release()
except Exception as e:
    print(f"[INFO] Hardware webcam not accessible ({e}). Running in Cloud/API mode.")

if not has_hardware_cam:
    print("[INFO] Cloud/API mode active: Image upload & Browser webcam streaming available.")

# Real-time state for live feed
latest_frame_bytes = None
latest_detection = {"detections": [], "status": "idle"}
state_lock = threading.Lock()
is_running = True

def run_inference_on_frame(frame: np.ndarray, yolo_model):
    """Run YOLO inference and annotate frame."""
    t0 = time.time()
    results = yolo_model(frame, verbose=False)
    inference_ms = round((time.time() - t0) * 1000, 1)

    annotated = frame.copy()
    detections = []
    top_label = "normal"
    top_conf = 0.0

    for r in results:
        # Classification model output
        if hasattr(r, "probs") and r.probs is not None:
            top_class_id = int(r.probs.top1)
            top_label = r.names[top_class_id]
            top_conf = round(float(r.probs.top1conf) * 100, 1)

            is_normal = top_label.lower() == "normal"
            color = (34, 197, 94) if is_normal else (0, 0, 239)  # Green / Red (BGR)
            label_text = f"{top_label.upper()}: {top_conf}%"

            cv2.rectangle(annotated, (10, 10), (360, 65), (15, 23, 42), -1)
            cv2.putText(annotated, label_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            if not is_normal:
                detections.append({"class": top_label, "confidence": top_conf})

        # Object Detection with Bounding Boxes
        elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = round(float(box.conf[0]) * 100, 1)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_normal = cls_name.lower() == "normal"
                color = (34, 197, 94) if is_normal else (0, 0, 239)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + 180, max(25, y1)), (15, 23, 42), -1)
                cv2.putText(annotated, f"{cls_name.upper()}: {conf}%", (x1 + 5, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if not is_normal:
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "box": [x1, y1, x2, y2]
                    })
            if detections:
                top_label = detections[0]["class"]
                top_conf = detections[0]["confidence"]

    return annotated, detections, top_label, top_conf, inference_ms

def hardware_camera_worker():
    """Background frame capture loop for local edge/hardware setups."""
    global latest_frame_bytes, latest_detection, is_running
    while is_running and has_hardware_cam:
        success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue

        with model_lock:
            m = active_yolo

        if m is not None:
            annotated, detections, top_label, top_conf, _ = run_inference_on_frame(frame, m)
        else:
            annotated = frame
            detections = []

        ret, buffer = cv2.imencode(".jpg", annotated)
        if ret:
            with state_lock:
                latest_frame_bytes = buffer.tobytes()
                latest_detection = {"detections": detections}

        time.sleep(0.02)

if has_hardware_cam:
    cam_thread = threading.Thread(target=hardware_camera_worker, daemon=True)
    cam_thread.start()

def generate_frames():
    """MJPEG stream generator for hardware camera or cloud standby frame."""
    while True:
        with state_lock:
            frame_bytes = latest_frame_bytes

        if frame_bytes is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        else:
            # Standby graphic when running in cloud without local webcam
            standby = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(standby, "CLOUD MODE ACTIVE", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(standby, "Use Browser Webcam or Image Upload tab", (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (148, 163, 184), 2)
            ret, buf = cv2.imencode(".jpg", standby)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(1.0)
        time.sleep(0.033)

# ----------------- REST API Endpoints -----------------

@app.get("/health")
def health_check():
    """Cloud health check probe."""
    return {
        "status": "healthy",
        "model": current_model_id,
        "hardware_camera": has_hardware_cam
    }

@app.get("/models")
def list_models():
    """List available YOLO models and current active model."""
    models_list = []
    for mid, info in AVAILABLE_MODELS.items():
        exists = os.path.exists(os.path.join(BASE_DIR, info["filename"]))
        models_list.append({
            "id": mid,
            "name": info["name"],
            "type": info["type"],
            "description": info["description"],
            "available": exists,
            "active": (mid == current_model_id)
        })
    return {"models": models_list, "active_model": current_model_id}

class SelectModelRequest(BaseModel):
    model_id: str

@app.post("/models/select")
def select_model(payload: SelectModelRequest):
    """Switch active YOLO model dynamically."""
    global current_model_id, active_yolo
    mid = payload.model_id
    if mid not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {mid}")
    with model_lock:
        try:
            active_yolo = load_yolo_model(mid)
            current_model_id = mid
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model {mid}: {e}")
    return {"status": "success", "active_model": current_model_id}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image file to run defect inspection.
    Returns detected defects and an annotated base64 JPEG image.
    """
    if not active_yolo:
        raise HTTPException(status_code=500, detail="No model loaded.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    with model_lock:
        annotated, detections, top_label, top_conf, inference_ms = run_inference_on_frame(img, active_yolo)

    ret, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8") if ret else ""

    return {
        "status": "success",
        "model": current_model_id,
        "is_defective": len(detections) > 0,
        "top_class": top_label,
        "top_confidence": top_conf,
        "detections": detections,
        "inference_ms": inference_ms,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }

class FramePayload(BaseModel):
    image: str  # Base64-encoded image frame from browser webcam

@app.post("/predict/frame")
def predict_frame(payload: FramePayload):
    """
    Inspect a video frame streamed from client's browser webcam.
    """
    if not active_yolo:
        raise HTTPException(status_code=500, detail="No model loaded.")

    try:
        data = payload.image
        if "," in data:
            data = data.split(",", 1)[1]
        decoded = base64.b64decode(data)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode frame.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame decode error: {e}")

    with model_lock:
        annotated, detections, top_label, top_conf, inference_ms = run_inference_on_frame(frame, active_yolo)

    return {
        "detections": detections,
        "status": top_label,
        "confidence": top_conf,
        "inference_ms": inference_ms
    }

@app.get("/video_feed")
def video_feed():
    """Live MJPEG video stream."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detect_status")
def detect_status():
    """Live telemetry polling endpoint."""
    with state_lock:
        return latest_detection

@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the web dashboard."""
    html_path = os.path.join(BASE_DIR, "test.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Defect Detector</title></head>
    <body style="background:#0f172a;color:#fff;font-family:sans-serif;padding:30px;text-align:center;">
        <h2>Industrial Defect Detector</h2>
        <p>API is running. Access <a href="/docs" style="color:#38bdf8;">/docs</a> for Swagger documentation.</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)