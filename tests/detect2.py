import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from datetime import datetime
import json

# Create 'detections' folder if it doesn't exist
output_folder = 'detections'
os.makedirs(output_folder, exist_ok=True)

# Format current date and time for filename
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Full output path
output_path = os.path.join(output_folder, f"{timestamp}.json")

# Load image
image_path = 'images/test3.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image.")
    exit()

# Load the pretrained YOLOv8 model (small version for speed)
model = YOLO('yolov8n.pt')  # 'n' = nano; other options: yolov8s.pt, yolov8m.pt, etc.

# Run detection
results = model(image)

# Extract result for the first image
result = results[0]

# Prepare data for JSON
detections = []

# Draw boxes on a copy of the image
image_with_boxes = image.copy()
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    conf = float(box.conf[0])  # Confidence score
    cls_id = int(box.cls[0])  # Class ID
    label = model.names[cls_id]  # Class name

    # Draw rectangle and label
    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_with_boxes, f'{label} {conf:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    detections.append({
        "label": label,
        "confidence": conf,
        "bbox": [x1, y1, x2, y2]
    })

# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

# Save JSON file to /detections/date-time.json
with open(output_path, 'w') as f:
    json.dump(detections, f, indent=4)

print(f"Detections saved to {output_path}")

# Show result
plt.imshow(image_rgb)
plt.title("Detected Objects")
plt.axis('off')
plt.show()
