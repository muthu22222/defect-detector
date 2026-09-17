import base64
import os
import time
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Defect Detection Live Stream & Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("defect_best.pt")

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
    print("[INFO] No hardware camera found on indices [0, 1, 2].")
    print("[INFO] Fix: Close all other video apps (Zoom, Teams, Camera). If you have multiple cameras, set CAMERA_INDEX=1 or CAMERA_INDEX=2.")
    print("[INFO] Cloud/Browser Mode: Browser webcam & image upload tabs are active.")

def generate_frames():
    """Video stream generator for local webcam or cloud standby graphic."""
    if not has_hardware_cam:
        # Render a clean cloud standby card instead of trying to open /dev/video0 in an infinite loop
        standby = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(standby, "CLOUD SERVER ACTIVE", (130, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(standby, "Render servers have no physical webcam.", (125, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)
        cv2.putText(standby, "Use Browser Webcam tab to inspect with your device!", (95, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1)
        ret, buf = cv2.imencode(".jpg", standby)
        frame_bytes = buf.tobytes()
        while True:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(1.0)

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue

        results = model(frame, verbose=False)

        for r in results:
            top_class_id = int(r.probs.top1)
            top_class_name = r.names[top_class_id]
            confidence = float(r.probs.top1conf) * 100

            color = (34, 197, 94) if top_class_name.lower() == "normal" else (0, 0, 239)
            label_text = f"{top_class_name.upper()}: {confidence:.1f}%"

            cv2.rectangle(frame, (10, 10), (340, 65), (15, 23, 42), -1)
            cv2.putText(frame, label_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Cloud Endpoint: Client Browser Webcam Frame Processing
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
        top_id = int(r.probs.top1)
        top_name = r.names[top_id]
        top_conf = round(float(r.probs.top1conf) * 100, 1)
        if top_name.lower() != "normal":
            detections.append({"class": top_name, "confidence": top_conf})

    return {
        "status": top_name,
        "confidence": top_conf,
        "detections": detections,
        "inference_ms": inference_ms
    }

# Cloud Endpoint: Image File Upload Inspection
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

    for r in results:
        top_id = int(r.probs.top1)
        top_name = r.names[top_id]
        top_conf = round(float(r.probs.top1conf) * 100, 1)

        is_normal = top_name.lower() == "normal"
        color = (34, 197, 94) if is_normal else (0, 0, 239)
        cv2.rectangle(frame, (10, 10), (340, 65), (15, 23, 42), -1)
        cv2.putText(frame, f"{top_name.upper()}: {top_conf}%", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        if not is_normal:
            detections.append({"class": top_name, "confidence": top_conf})

    ret, buf = cv2.imencode(".jpg", frame)
    b64_img = base64.b64encode(buf).decode("utf-8") if ret else ""

    return {
        "top_class": top_name,
        "top_confidence": top_conf,
        "detections": detections,
        "inference_ms": inference_ms,
        "annotated_image": f"data:image/jpeg;base64,{b64_img}"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "hardware_camera": has_hardware_cam,
        "camera_index": camera_index
    }

@app.get("/detect_status")
def detect_status():
    return {"detections": []}

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(BASE_DIR, "test.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Defect Detection</title></head>
    <body style="background:#0f172a;color:#fff;text-align:center;padding:24px;">
        <h2>Defect Detection API Active</h2>
        <a href="/docs" style="color:#38bdf8;">View Swagger API Docs</a>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
