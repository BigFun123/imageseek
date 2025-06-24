import cv2
from ultralytics import YOLO
import trafficlights

# people, driving, object
mode = "people"


# Load the pretrained YOLOv8 model (small version for speed)
model = YOLO('models/yolov8s.pt')  # 'n' = nano; other options: yolov8s.pt, yolov8m.pt, etc.
model = model.to('cuda')  # Move model to GPU if available
boxcolor = (240, 255, 0)  # Color for bounding boxes (BGR format)
personcolor = (255, 0, 255)  # Color for person bounding boxes (BGR format)
carcolor = (0, 0, 255)  # Color for car bounding boxes (BGR format) 

results = None  # Initialize results variable
result = None  # Initialize result variable
detections = []  # Initialize detections list
filterout = []  # Initialize filterout list for unwanted objects
if mode == "people":
    filterout = ["car", "bus", "truck", "bicycle", "motorcycle", "train", "airplane", "boat", "traffic light", "traffic lights", "person", "tie", "cup"]
if mode == "driving":    
    filterout = ["tv", "cell_phone", "laptop", "mouse", "keyboard"]

def detect_objects(smallframe, scale):
    global results, result, model, boxcolor, personcolor, carcolor
    # Run detection
    results = model(smallframe, verbose=False, device=0)

    # Extract result for the first image
    result = results[0]
    
    for box in result.boxes:
        x1, y1, x2, y2 = [int(coord * (1/scale)) for coord in box.xyxy[0]]  # Bounding box coordinates scaled
        conf = float(box.conf[0])  # Confidence score
        cls_id = int(box.cls[0])  # Class ID
        label = model.names[cls_id]  # Class name
        if (label == 'traffic light' or label == 'traffic lights'):            
            trafficlights.handleTrafficLights(smallframe, box)
  
        
def draw(frame, scale):
    global detections
    # Draw bounding boxes on the original frame
    #results = model(frame, verbose=False, device=0)
    #result = results[0]
    
    trafficlights.draw(frame, scale)  # Draw traffic lights first    

    for box in result.boxes:
        x1, y1, x2, y2 = [int(coord * (1/scale)) for coord in box.xyxy[0]]  # Bounding box coordinates scaled
        conf = float(box.conf[0])  # Confidence score
        cls_id = int(box.cls[0])  # Class ID
        label = model.names[cls_id]  # Class name
        
        

        color = boxcolor  # Default color for bounding boxes
        if label == 'person':
            color = personcolor
        elif label == 'car':
            color = carcolor

        # Draw rectangle and label
        corner_length = 5  # Length of the corner lines
        thickness = 2

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)

        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)

        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)

        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
        
        # Draw a transparent filled rectangle for the bounding box
        # check if the label is in the filter list
        if label not in filterout:            
            overlay = frame.copy()
            alpha = 0.2  # Transparency factor (0.0 - 1.0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })