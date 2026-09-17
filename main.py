import cv2
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

# This is the "app" variable Uvicorn is looking for
app = FastAPI(title="Defect Detection Live Stream")

model = YOLO("defect_best.pt")
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, verbose=False)

        for r in results:
            top_class_id = r.probs.top1
            top_class_name = r.names[top_class_id]
            confidence = float(r.probs.top1conf) * 100

            color = (0, 255, 0) if top_class_name.lower() == "normal" else (0, 0, 255)
            label_text = f"{top_class_name.upper()}: {confidence:.1f}%"

            cv2.rectangle(frame, (10, 10), (340, 65), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Machine Defect Detection Feed</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔍</text></svg>">
        <style>
            body { background: #0f172a; color: #f8fafc; font-family: sans-serif; text-align: center; padding: 24px; }
            img { max-width: 90%; border: 2px solid #334155; border-radius: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            h2 { margin-bottom: 16px; font-weight: 600; }
        </style>
    </head>
    <body>
        <h2>Live Machine Defect Detection Feed</h2>
        <img src="/video_feed" alt="Camera Stream" />
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

