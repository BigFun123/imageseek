import cv2
import numpy as np
import mss
import objectdetection
import videwrite
import facedetection
from datetime import datetime

# === CONFIGURATION ===
use_camera = False  # Set to False to use screen capture
#capture_area = (1930, 100, 1930+400, 580)  # Only used if use_camera = False
scale = 0.75  # Resize factor for performance (smaller = faster)
process_every_n_frames = 1  # Only process every Nth frame
faces_dir = "faces"  # Folder with known faces
frame_count = 0  # Frame counter for processing control
left_offset = 200  # Offset for cropping the screen capture
top_offset = 200  # Offset for cropping the screen capture
captureToFile = False
videout = None  # Video writer object

# === INITIALIZE VIDEO ===
if use_camera:
    cap = cv2.VideoCapture(0)
else:
    # Create a persistent screen grabber
    sct = mss.mss()    
    # Example: Capture from monitor 2 (second monitor)
    monitor_index = 2  # 1 = primary, 2 = second, etc.
    monitor = sct.monitors[monitor_index]
    
# Optional: crop inside monitor area
capture_area = {
    "top": monitor["top"] +  top_offset,
    "left": monitor["left"] + left_offset,
    "width": 1024,
    "height": 768
}

names = []
face_locations = []
face_encodings = []
# Prepare data for JSON
detections = []

facedetection.setup()  # Load known faces from the 'faces' directory

while True:
    capture_area["top"] =  monitor["top"] +  top_offset
    capture_area["left"] = monitor["left"] + left_offset    
    
    if use_camera:
        success, frame = cap.read()
    else:        
        screen = np.array(sct.grab(capture_area))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        success = True

    if not success:
        break
    
    #frame += 1
    
    # Resize frame for faster face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small_frame = small_frame[:, :, ::-1]
    if frame_count % process_every_n_frames == 0:    
        objectdetection.detect_objects(small_frame, scale)
        facedetection.detectfaces(rgb_small_frame)
    

    
    key = cv2.waitKey(1)
    #print(key)
    # Print shortcut keys at the top
    if frame_count == 0:
        print("Shortcut keys:")
        print("  q - Quit")
        print("  c - Toggle video capture to file")
        print("  s - Save current frame faces")
        print("  r - Reload known faces")
        print("  [ / ] - Move capture area left/right")
        print("  { / } - Move capture area up/down")

    if key == ord("q"):
        break
    if key == ord("c"):
        captureToFile = not captureToFile
        print(f"Capture to file: {'ON' if captureToFile else 'OFF'}")
        
        if videout is None and captureToFile:
            videout = videwrite.setup(20, capture_area["width"], capture_area["height"])        
        else:
            videout.release()        
            videout = None        
    if key == ord("s"):
        # Save current frame as an image
        facedetection.save_faces(frame)        
    if key == ord("r"):        
        facedetection.setup()  # Reload known faces from the 'faces' directory    
    if key == ord("["):
        left_offset -= 50
    if key == ord("]"):        
        left_offset += 50    
    if key == ord("{"):        
        top_offset -= 50    
    if key == ord("}"):        
        top_offset += 50
    
        

    objectdetection.draw(frame, scale)
    temp = facedetection.draw(frame, scale)
    if temp is not None:
        frame = temp

    if captureToFile:
        videout.write(frame)
    cv2.imshow("Face Recognition", frame)
    
    frame_count += 1

if use_camera:
    cap.release()
if videout is not None:    
    videout.release()    
cv2.destroyAllWindows()
