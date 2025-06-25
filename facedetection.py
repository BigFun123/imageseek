from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
import tools

detected = []
face_cache = []
#scale = 1.0  # Scale factor for bounding box coordinates
remove_pii = False  # Remove PII from the code comments

known_faces = {}  # Dictionary to store known face embeddings
threshold = 0.4

# Load model (includes detector and recognition model)
# buffalo_l is a lightweight model suitable for real-time applications
# You can also use 'buffalo_s' for a smaller model or 'buffalo_m'
model = FaceAnalysis(name='buffalo_m', 
        #root='models',
        providers=['CUDAExecutionProvider'], 
        allowed_modules=['detection', 'recognition', 'genderage'])
model.prepare(ctx_id=0, det_thresh=threshold)  # 0 = use GPU

def normalize(embedding):
    return embedding / np.linalg.norm(embedding)


def setup():
    global known_faces
    print("Loading known faces...")
    # Load known faces (e.g., from faces/ folder)
    known_faces = {}
    for root, dirs, files in os.walk('faces'):
        for filename in files:
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
                img = cv2.imread(os.path.join(root, filename))
                print(filename)
                faces = model.get(img)
                if faces:
                    embedding = normalize(faces[0].embedding)
                    # Get relative path from 'faces' and use as name (replace os.sep with '_')
                    rel_path = os.path.relpath(os.path.join(root, filename), 'faces')
                    name = os.path.splitext(rel_path)[0].replace(os.sep, '_')
                    known_faces[name] = embedding
                else:
                    print(f"No face detected in {filename}. Skipping.")

# grab the cropped face images and save them to the temp folder with timestamp
def save_faces(frame):
    global known_faces
    faces = model.get(frame)
    if not faces:
        return
    
    faces.sort(key=lambda x: x.bbox[2] - x.bbox[0] + x.bbox[3] - x.bbox[1], reverse=True)  # Sort by size
    largest_face = faces[0]  # Get the largest face
    bbox = largest_face.bbox.astype(int)
    # expand the bbox by 10% to include more of the face
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    bbox[0] = max(0, int(bbox[0] - width * 0.3))
    bbox[1] = max(0, int(bbox[1] - height * 0.3))
    bbox[2] = min(frame.shape[1], int(bbox[2] + width * 0.3))
    bbox[3] = min(frame.shape[0], int(bbox[3] + height * 0.3))
    # Ensure the bounding box is within the frame dimensions
    bbox = np.clip(bbox, 0, [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                
    # Crop the face region
    cropped_face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Save the cropped face image
    timestamp_str = tools.get_timestamp()
    filename = f"temp/face_{timestamp_str}.jpg"
    cv2.imwrite(filename, cropped_face)

# Load target image (e.g., a frame from a video)
def detectfaces(smallframe):
    #frame = cv2.imread('video_faces/frame.jpg')
    global detected
    detected = model.get(smallframe)

def draw(frame, scale):
    # Compare and label
    global detected
    global known_faces
    
    
    for face in detected:
        emb = normalize(face.embedding)
    
        # Initialize default values    
        name = "Unknown"
        best_score = float("inf")
        sex = "?"
        age = "?"
        score = 0.0

        for known_name, known_emb in known_faces.items():
            score = np.linalg.norm(emb - known_emb)
            #print(f"{known_name}: {score}")
            if score < 0.9 and score < best_score:
                best_score = score
                name = known_name
                sex = face.sex
                age = face.age

        # Scale the bounding box coordinates individually
        box = face.bbox.astype(int)
        box = np.array([int(coord * (1/scale)) for coord in box])        
        #cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 1)
        bounds =  (box[0], box[1], box[2], box[3])
        tools.drawOutlineBox(frame, bounds, (55, 255, 55), 1)
        if (label := name) == "Unknown" and remove_pii:
            # draw an empty rectangle if no known face is detected
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (55, 55, 55), -1)
            
            
        label = f"{name} {age} {sex} {best_score:.1f}"
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # Draw filled rectangle under the text
        cv2.rectangle(frame, 
                  (box[0], box[1] - 7 - text_height - baseline), 
                  (box[0] + text_width, box[1] - 7), 
                  (0, 255, 0), 
                  thickness=cv2.FILLED)
        # Draw the text label
        #cv2.putText(frame, label, (box[0], box[1] - 10),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y = max(0, box[1] - 20)
        frame = tools.draw_unicode_text(frame, label, (box[0], y), font_path="ARIAL.TTF", font_size=14)
        
    return frame
    
