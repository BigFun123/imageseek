import cv2
import numpy as np

inset_ratio = 0.1

# Red
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Green
lower_green = np.array([10, 50, 50])
upper_green = np.array([90, 255, 255])

# Stores detected traffic lights to draw later
detections = []

def handleTrafficLights(frame, trafficLightbox):
    global lower_red1, upper_red1, lower_red2, upper_red2
    x1, y1, x2, y2 = map(int, trafficLightbox.xyxy[0].cpu().numpy())
    # Calculate inset margins
    width = x2 - x1
    height = y2 - y1
    dx = int(width * inset_ratio)
    dy = int(height * inset_ratio)

    # Apply inset
    x1_inset = max(x1 + dx, 0)
    y1_inset = max(y1 + dy, 0)
    x2_inset = min(x2 - dx, frame.shape[1])
    y2_inset = min(y2 - dy, frame.shape[0])

    # Crop and analyze color
    cropped = frame[y1_inset:y2_inset, x1_inset:x2_inset]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    threshold = 20  # Minimum number of pixels to consider a color dominant
    
    color = "Unknown"
    if red_pixels > green_pixels and red_pixels > threshold:
        color = "Red"
    elif green_pixels > red_pixels and green_pixels > threshold:
        color = "Green"        
    
    detections.append((x1, y1, x2, y2, color))  # Store for later


def draw(frame, scale=1.0):
    global detections
    for x1, y1, x2, y2, color in detections:
        # Scale coordinates back up if needed
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)

        color_map = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Unknown": (0, 255, 255)
        }
        box_color = color_map.get(color, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        if color != "Unknown":
            # Draw a filled rectangle for the label background
            cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), box_color, -1)
            cv2.putText(frame, color, (x1, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    detections.clear()  # Clear after drawing

    