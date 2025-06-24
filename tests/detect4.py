import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from datetime import datetime
import json

# Make detections folder if it doesn't exist
output_dir = "detections"
os.makedirs(output_dir, exist_ok=True)

# Create timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
json_path = os.path.join(output_dir, f"{timestamp}.json")
image_output_path = os.path.join(output_dir, f"{timestamp}.jpg")

# Load image
image_path = 'images/nude.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image.")
    exit()

# Load the pretrained YOLOv8 model (small version for speed)
model = YOLO('models/yolov8n.pt')  # 'n' = nano; other options: yolov8s.pt, yolov8m.pt, etc.
# Load face-specific YOLOv8 model
face_model = YOLO('models/yolov8n-face.pt')

# Run face detection
face_results = face_model(image)
face_result = face_results[0]

# Run detection
results = model(image)

# Extract result for the first image
result = results[0]

# Prepare data for JSON
detections = []


# Load the Haar Cascade face detector
#face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Convert image to grayscale for face detection
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
#faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


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

# # Draw face rectangles
# for (x, y, w, h) in faces:
#     cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     detections.append({
#         "label": "face",
#         "confidence": None,  # Haar cascade doesn't give confidence
#         "bbox": [int(x), int(y), int(x + w), int(y + h)]
#     })
    
# Draw YOLO face boxes
for box in face_result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    label = "face"

    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(image_with_boxes, f'{label} {conf:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Add to detections
    detections.append({
        "label": label,
        "confidence": conf,
        "bbox": [x1, y1, x2, y2]
    })    
    
    
# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

# Save JSON file to /detections/date-time.json
with open(json_path, 'w') as f:
    json.dump(detections, f, indent=4)

print(f"Detections saved to {json_path}")




# Show result
plt.imshow(image_rgb)
plt.title("Detected Objects")
plt.axis('off')
plt.show()

# Save JSON
with open(json_path, "w") as f:
    json.dump(detections, f, indent=2)

# Save image with rectangles
cv2.imwrite(image_output_path, image_with_boxes)

print(f"Saved JSON: {json_path}")
print(f"Saved Image: {image_output_path}")
