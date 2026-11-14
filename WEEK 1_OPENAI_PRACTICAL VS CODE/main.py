"""
VisionPort: Intelligent Vehicle Detection and Traffic Monitoring
Tasks 1-15 implementation.

Run:
    python -m pip install -r requirements.txt
    python -m uvicorn main:app --host 127.0.0.1 --port 8000

Notes:
- YOLOv8n COCO classes include car, motorcycle, bus, truck, bicycle and traffic light.
- Speed/waiting-time values are estimates and require camera calibration.
- Density/congestion thresholds are configurable demonstration heuristics.
"""
import os
import cv2
import json
import time
import uuid
import shutil
import threading
import numpy as np
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
FRAMES = BASE / "extracted_frames"
OUTPUTS.mkdir(exist_ok=True)
FRAMES.mkdir(exist_ok=True)

MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF = float(os.getenv("CONFIDENCE", "0.25"))
VIDEO_STEP = max(1, int(os.getenv("VIDEO_FRAME_STEP", "2")))
VIDEO_IMGSZ = int(os.getenv("VIDEO_IMGSZ", "480"))
# Set this after calibrating a fixed camera. A single scale is only approximate.
PIXELS_PER_METER = float(os.getenv("PIXELS_PER_METER", "20"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# COCO model class labels used by YOLOv8n.
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
app = FastAPI(title="VisionPort - Traffic Detection and Monitoring", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")
model = YOLO(MODEL_PATH)
model_lock = threading.Lock()


def decode_image(data: bytes):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image file.")
    return image


def density_for(count):
    if count <= 5:
        return "Low"
    if count <= 15:
        return "Medium"
    return "High"


def traffic_light_colour(frame, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = {
        "Red": (cv2.inRange(hsv, (0, 90, 70), (10, 255, 255)) |
                cv2.inRange(hsv, (170, 90, 70), (180, 255, 255))),
        "Yellow": cv2.inRange(hsv, (15, 80, 90), (40, 255, 255)),
        "Green": cv2.inRange(hsv, (40, 60, 50), (90, 255, 255))
    }
    scores = {k: cv2.countNonZero(v) for k, v in masks.items()}
    winner = max(scores, key=scores.get)
    return winner if scores[winner] >= 2 else "Unknown"


def analyze_result(result, frame):
    names = result.names
    vehicles, lights = [], []
    boxes = result.boxes
    if boxes is not None:
        for b in boxes:
            cls = int(b.cls[0])
            label = names.get(cls, str(cls))
            conf = float(b.conf[0])
            coords = b.xyxy[0].cpu().numpy().tolist()
            if label in VEHICLE_CLASSES:
                vehicles.append({
                    "class": label, "confidence": round(conf, 3),
                    "box": [round(float(x), 1) for x in coords]
                })
            elif label == "traffic light":
                lights.append({
                    "status": traffic_light_colour(frame, coords),
                    "confidence": conf, "box": coords
                })
    signal = "Unknown"
    if lights:
        signal = max(lights, key=lambda x: x["confidence"])["status"]
    return vehicles, lights, signal


def annotate(frame, result, count, density, signal, congestion=None):
    out = result.plot()
    cv2.putText(out, f"Vehicles: {count} | Density: {density}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(out, f"Signal: {signal}", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
    if congestion:
        cv2.putText(out, f"Congestion: {congestion}", (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)
    return out


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    return {
        "status": "online", "model": MODEL_PATH,
        "video_frame_step": VIDEO_STEP, "video_imgsz": VIDEO_IMGSZ
    }


# Task 2: capture a still image from the camera connected to the server.
@app.post("/camera/capture")
def capture_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise HTTPException(503, "Camera could not be opened. Check CAMERA_INDEX and camera permissions.")
    try:
        ok, frame = cap.read()
        if not ok:
            raise HTTPException(503, "Camera opened but no frame was received.")
        filename = f"camera_{uuid.uuid4().hex[:8]}.jpg"
        path = OUTPUTS / filename
        cv2.imwrite(str(path), frame)
        return {"image_url": f"/outputs/{filename}", "message": "Camera image captured."}
    finally:
        cap.release()


# Tasks 3 and 8: image decode, inference, annotated image + labels/confidences.
@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    frame = decode_image(await file.read())
    start = time.perf_counter()
    with model_lock:
        result = model.predict(frame, conf=CONF, imgsz=640, verbose=False)[0]
    elapsed = (time.perf_counter() - start) * 1000
    vehicles, lights, signal = analyze_result(result, frame)
    count = len(vehicles)
    density = density_for(count)
    output = annotate(frame, result, count, density, signal)
    filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(OUTPUTS / filename), output)
    return {
        "annotated_image": f"/outputs/{filename}",
        "vehicle_count": count, "density": density,
        "signal_status": signal, "inference_ms": round(elapsed, 1),
        "detections": vehicles, "traffic_lights": lights
    }


# Tasks 4-15: video playback, frame extraction, YOLO, count/density,
# congestion, ByteTrack IDs, approximate speed, signal and waiting-time report.
@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "traffic.mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
        raise HTTPException(400, "Use MP4, AVI, MOV, MKV or M4V video.")
    job = uuid.uuid4().hex[:10]
    input_path = OUTPUTS / f"{job}_input{suffix}"
    output_path = OUTPUTS / f"{job}_annotated.mp4"
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read this video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "Video dimensions are invalid.")

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        input_path.unlink(missing_ok=True)
        raise HTTPException(500, "Could not create output MP4. Try another video codec/format.")

    frame_no = 0
    frames_total = 0
    frames_analyzed = 0
    counts = []
    max_count = 0
    class_totals = defaultdict(int)
    signal_votes = defaultdict(int)
    last_annotated = None
    previous_centres = {}
    stopped_since = {}
    longest_stops = defaultdict(float)
    speed_values = []
    frame_interval = max(1, int(round(fps)))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_total += 1

            if frame_no % VIDEO_STEP == 0:
                with model_lock:
                    result = model.track(
                        frame, persist=True, tracker="bytetrack.yaml",
                        conf=CONF, imgsz=VIDEO_IMGSZ, verbose=False
                    )[0]
                vehicles, lights, signal = analyze_result(result, frame)
                count = len(vehicles)
                counts.append(count)
                max_count = max(max_count, count)
                frames_analyzed += 1
                for item in vehicles:
                    class_totals[item["class"]] += 1
                if signal != "Unknown":
                    signal_votes[signal] += 1

                # Tracking IDs supplied by ByteTrack.
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    coords_list = boxes.xyxy.cpu().numpy()
                    for track_id, coords in zip(ids, coords_list):
                        cx = float((coords[0] + coords[2]) / 2)
                        cy = float((coords[1] + coords[3]) / 2)
                        now = frame_no / fps
                        if track_id in previous_centres:
                            old_x, old_y, old_t = previous_centres[track_id]
                            dt = max(now - old_t, 1.0 / fps)
                            # Approximate only; requires a calibrated fixed camera.
                            metres = np.hypot(cx - old_x, cy - old_y) / max(PIXELS_PER_METER, 0.01)
                            kmh = min((metres / dt) * 3.6, 180.0)
                            speed_values.append(kmh)
                            if kmh < 3:
                                stopped_since.setdefault(track_id, now)
                                longest_stops[track_id] = max(
                                    longest_stops[track_id], now - stopped_since[track_id]
                                )
                            else:
                                stopped_since.pop(track_id, None)
                        previous_centres[track_id] = (cx, cy, now)

                avg_now = float(np.mean(counts)) if counts else 0
                congestion_now = ("High" if count > 15 else
                                  "Medium" if count > 5 else "Low")
                last_annotated = annotate(
                    frame, result, count, density_for(count), signal, congestion_now
                )

            # Preserve original frame rate/duration; skipped frames reuse last annotated frame.
            writer.write(last_annotated if last_annotated is not None else frame)

            # Task 5: save one original frame per second.
            if frame_no % frame_interval == 0:
                cv2.imwrite(str(FRAMES / f"{job}_frame_{frame_no:06d}.jpg"), frame)
            frame_no += 1
    finally:
        cap.release()
        writer.release()
        input_path.unlink(missing_ok=True)

    if frames_total == 0:
        output_path.unlink(missing_ok=True)
        raise HTTPException(400, "No readable frames found in video.")

    avg_count = float(np.mean(counts)) if counts else 0.0
    avg_speed = float(np.mean(speed_values)) if speed_values else 0.0
    wait_list = list(longest_stops.values())
    avg_wait = float(np.mean(wait_list)) if wait_list else 0.0
    density = density_for(round(avg_count))
    congestion = ("High" if avg_count > 15 else
                  "Medium" if avg_count > 5 else "Low")
    signal = max(signal_votes, key=signal_votes.get) if signal_votes else "Unknown"

    return {
        "job_id": job,
        "processed_video": f"/outputs/{output_path.name}",
        "frames_processed": frames_total,
        "frames_analyzed_by_yolo": frames_analyzed,
        "frame_step": VIDEO_STEP,
        "vehicle_count_average": round(avg_count, 2),
        "maximum_vehicle_count": max_count,
        "density": density,
        "congestion": congestion,
        "congestion_alert": f"{congestion} traffic congestion" if congestion != "Low" else "No significant congestion",
        "vehicle_detections_by_class": dict(class_totals),
        "average_estimated_speed_kmh": round(avg_speed, 2),
        "estimated_waiting_time_seconds": round(avg_wait, 2),
        "traffic_signal_status": signal,
        "extracted_frames_saved": int(np.ceil(frames_total / frame_interval)),
        "calibration_note": "Speed and waiting time are rough estimates. Calibrate PIXELS_PER_METER and use a fixed camera for meaningful values."
    }


@app.get("/frames")
def list_extracted_frames():
    files = sorted(FRAMES.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"count": len(files), "frames": [p.name for p in files[:200]]}
