import base64
import os
import time
import cv2
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Industrial Defect Vision AI",
    description="Automated defect detection and classification API powered by YOLOv8 and FastAPI.",
    version="1.0.0"
)

# Enable CORS for cross-origin client apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models dictionary & active model
MODELS = {
    "defect_best": YOLO(os.path.join(BASE_DIR, "defect_best.pt")),
    "best": YOLO(os.path.join(BASE_DIR, "best.pt"))
}
active_model_id = "defect_best"
model = MODELS[active_model_id]

# Hardware Camera Initialization (with auto-detection for indices 0, 1, 2 and CAMERA_INDEX override)
camera = None
camera_index = None
has_hardware_cam = False

def init_webcam():
    """Attempt to open hardware camera at configured or auto-detected index (0, 1, 2)."""
    env_index = os.environ.get("CAMERA_INDEX")
    if env_index is not None and env_index.strip().isdigit():
        candidate_indices = [int(env_index.strip())]
    else:
        candidate_indices = [0, 1, 2]

    is_windows = os.name == "nt"

    for idx in candidate_indices:
        backends = [cv2.CAP_DSHOW, None] if is_windows else [None]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        backend_name = "CAP_DSHOW" if backend == cv2.CAP_DSHOW else "Default"
                        print(f"[INFO] Hardware camera successfully connected on index {idx} ({backend_name}).")
                        return cap, idx
                    cap.release()
                else:
                    cap.release()
            except Exception:
                pass
    return None, None

camera, camera_index = init_webcam()
has_hardware_cam = camera is not None

if not has_hardware_cam:
    print("[INFO] No hardware camera attached (Normal for Cloud Deployment).")
    print("[INFO] Browser webcam (/predict/frame) and image upload (/predict/image) are active.")

# Telemetry for server stream
latest_detections = []

def generate_frames():
    """Video stream generator for local webcam or cloud standby graphic."""
    global latest_detections
    if not has_hardware_cam:
        standby = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(standby, "CLOUD SERVER ACTIVE", (130, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(standby, "Cloud container has no physical webcam.", (115, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)
        cv2.putText(standby, "Use Browser Webcam tab to inspect with your device!", (85, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1)
        ret, buf = cv2.imencode(".jpg", standby)
        frame_bytes = buf.tobytes()
        while True:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(1.0)

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue

        results = model(frame, verbose=False)
        frame_detections = []

        for r in results:
            if hasattr(r, "probs") and r.probs is not None:
                # Classification model
                top_id = int(r.probs.top1)
                top_name = r.names[top_id]
                confidence = float(r.probs.top1conf) * 100
                color = (34, 197, 94) if top_name.lower() == "normal" else (0, 0, 239)
                label_text = f"{top_name.upper()}: {confidence:.1f}%"
                cv2.rectangle(frame, (10, 10), (340, 65), (15, 23, 42), -1)
                cv2.putText(frame, label_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                if top_name.lower() != "normal":
                    frame_detections.append({"class": top_name, "confidence": round(confidence, 1)})
            elif hasattr(r, "boxes") and r.boxes is not None:
                # Detection model (bounding boxes)
                frame = r.plot()
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names[cls_id]
                    conf = round(float(box.conf[0]) * 100, 1)
                    if cls_name.lower() != "normal":
                        frame_detections.append({"class": cls_name, "confidence": conf})

        latest_detections = frame_detections
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "active_model": active_model_id,
        "hardware_camera": has_hardware_cam,
        "camera_index": camera_index
    }

@app.get("/detect_status")
def detect_status():
    return {"detections": latest_detections}

@app.get("/models")
def list_models():
    return {
        "active": active_model_id,
        "available": list(MODELS.keys())
    }

class ModelSelectPayload(BaseModel):
    model_id: str

@app.post("/models/select")
def select_model(payload: ModelSelectPayload):
    global model, active_model_id
    if payload.model_id in MODELS:
        active_model_id = payload.model_id
        model = MODELS[active_model_id]
        return {"status": "success", "active_model": active_model_id}
    raise HTTPException(status_code=400, detail=f"Model {payload.model_id} not found.")

class FramePayload(BaseModel):
    image: str

@app.post("/predict/frame")
def predict_frame(payload: FramePayload):
    try:
        data = payload.image
        if "," in data:
            data = data.split(",", 1)[1]
        decoded = base64.b64decode(data)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decode error: {e}")

    t0 = time.time()
    results = model(frame, verbose=False)
    inference_ms = round((time.time() - t0) * 1000, 1)

    top_name = "normal"
    top_conf = 0.0
    detections = []

    for r in results:
        if hasattr(r, "probs") and r.probs is not None:
            top_id = int(r.probs.top1)
            top_name = r.names[top_id]
            top_conf = round(float(r.probs.top1conf) * 100, 1)
            if top_name.lower() != "normal":
                detections.append({"class": top_name, "confidence": top_conf})
        elif hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = round(float(box.conf[0]) * 100, 1)
                detections.append({"class": cls_name, "confidence": conf})
                if conf > top_conf:
                    top_conf = conf
                    top_name = cls_name

    return {
        "status": top_name,
        "confidence": top_conf,
        "detections": detections,
        "inference_ms": inference_ms
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    t0 = time.time()
    results = model(frame, verbose=False)
    inference_ms = round((time.time() - t0) * 1000, 1)

    top_name = "normal"
    top_conf = 0.0
    detections = []

    annotated = frame.copy()
    for r in results:
        if hasattr(r, "probs") and r.probs is not None:
            top_id = int(r.probs.top1)
            top_name = r.names[top_id]
            top_conf = round(float(r.probs.top1conf) * 100, 1)
            color = (34, 197, 94) if top_name.lower() == "normal" else (0, 0, 239)
            cv2.rectangle(annotated, (10, 10), (340, 65), (15, 23, 42), -1)
            cv2.putText(annotated, f"{top_name.upper()}: {top_conf}%", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            if top_name.lower() != "normal":
                detections.append({"class": top_name, "confidence": top_conf})
        elif hasattr(r, "boxes") and r.boxes is not None:
            annotated = r.plot()
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = round(float(box.conf[0]) * 100, 1)
                detections.append({"class": cls_name, "confidence": conf})
                if conf > top_conf:
                    top_conf = conf
                    top_name = cls_name

    ret, buf = cv2.imencode(".jpg", annotated)
    b64_img = base64.b64encode(buf).decode("utf-8") if ret else ""

    return {
        "top_class": top_name,
        "top_confidence": top_conf,
        "detections": detections,
        "inference_ms": inference_ms,
        "annotated_image": f"data:image/jpeg;base64,{b64_img}"
    }

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(BASE_DIR, "test.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Defect Detection API Active</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔍</text></svg>">
        <style>
            body { background: #0f172a; color: #f8fafc; font-family: sans-serif; text-align: center; padding: 40px; }
            a { color: #38bdf8; text-decoration: none; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>Defect Detection API is Online</h2>
        <p><a href="/docs">Open Interactive Swagger API Docs &rarr;</a></p>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
