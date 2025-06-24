import cv2
import dlib
import os

# Setup
input_dir = "faces"
output_dir = "faces/cropped"
os.makedirs(output_dir, exist_ok=True)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)

    if img is None:
        print(f"Could not read {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        print(f"No face found in {filename}")
        continue

    # Use the first detected face
    face = faces[0]
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

    # Add padding
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.shape[1], x2 + padding)
    y2 = min(img.shape[0], y2 + padding)

    # Crop and resize
    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (150, 150))

    # Save
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, resized)
    print(f"Cropped face saved to {out_path}")
