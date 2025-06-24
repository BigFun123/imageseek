import cv2
from ultralytics import YOLO
from datetime import datetime
import json
import os

# Load models
object_model = YOLO("models/yolov8n.pt")
face_model = YOLO("models/yolov8n-face.pt")

# Open video
video_path = "videos/trump_hi.mp4"  # Replace with your input video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Timestamped output
timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
output_folder = f"detections/{timestamp_str}"
os.makedirs(output_folder, exist_ok=True)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_out_path = os.path.join(output_folder, f"{timestamp_str}.mp4")
out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

# JSONL file
jsonl_path = os.path.join(output_folder, f"{timestamp_str}.jsonl")
output_file = open(jsonl_path, "w")

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    timestamp = round(frame_number / fps, 2)
    detections = []

    # Object detection
    obj_results = object_model(frame)[0]
    for box in obj_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = object_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        detections.append({
            "type": "object",
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2]
        })
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Face detection
    face_results = face_model(frame)[0]
    for box in face_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append({
            "type": "face",
            "label": "face",
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2]
        })
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"face {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Write frame to video
    out.write(frame)

    # Write JSONL entry
    output_file.write(json.dumps({
        "frame": frame_number,
        "timestamp": timestamp,
        "detections": detections
    }) + "\n")

    # Optional live view
    cv2.imshow("Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
output_file.close()
cv2.destroyAllWindows()
