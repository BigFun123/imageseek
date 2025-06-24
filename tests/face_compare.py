import cv2
import dlib
import numpy as np
import os

# Load models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

# Load known face encodings
def load_face_encodings(face_dir):
    encodings = []
    names = []
    for filename in os.listdir(face_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(face_dir, filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector(gray)
            if faces:
                shape = shape_predictor(gray, faces[0])
                face_descriptor = face_rec_model.compute_face_descriptor(gray, shape)
                encodings.append(np.array(face_descriptor))
                names.append(os.path.splitext(filename)[0])
    return encodings, names

known_encodings, known_names = load_face_encodings("faces/cropped")

# Process video
video_path = "videos/video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)
    for face in faces:
        shape = shape_predictor(rgb, face)
        descriptor = np.array(face_rec_model.compute_face_descriptor(rgb, shape))

        # Compare to known faces
        distances = np.linalg.norm(known_encodings - descriptor, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] < 0.6:
            name = known_names[min_idx]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,0), 2)
        cv2.putText(frame, name, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Face Match", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
