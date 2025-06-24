import cv2
import dlib
import numpy as np

# === Config ===
video_path = "videos/video.mp4"
landmark_path = "models/shape_predictor_68_face_landmarks.dat"

scale_factor = 1.4           # Enlargement factor for head
top_margin_factor = 0.25     # How much more headroom to add above the detected face
blur_amount = 51             # Must be odd and >= 3, controls softness of edge mask
blur_sigma = 50              # Strength of the Gaussian blur
show_markers = False         # Toggle with 'm'

# === Setup ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_path)
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Expand face rect upward
        top_margin = int(top_margin_factor * h)
        new_y = max(y - top_margin, 0)
        new_h = h + (y - new_y)

        face_roi = frame[new_y:new_y+new_h, x:x+w]
        enlarged_w, enlarged_h = int(w * scale_factor), int(new_h * scale_factor)
        resized_face = cv2.resize(face_roi, (enlarged_w, enlarged_h))

        # Elliptical mask
        mask = np.zeros(resized_face.shape[:2], dtype=np.uint8)
        center = (enlarged_w // 2, enlarged_h // 2)
        axes = (int(enlarged_w * 0.5), int(enlarged_h * 0.7))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Apply Gaussian blur to the mask
        if blur_amount % 2 == 0:
            blur_amount += 1  # Must be odd
        blurred_mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), blur_sigma)

        # Compute paste position
        cx, cy = x + w // 2, new_y + new_h // 2
        paste_x = max(cx - enlarged_w // 2, 0)
        paste_y = max(cy - enlarged_h // 2, 0)
        paste_x2 = min(paste_x + enlarged_w, frame.shape[1])
        paste_y2 = min(paste_y + enlarged_h, frame.shape[0])

        # Clip
        clip_w = paste_x2 - paste_x
        clip_h = paste_y2 - paste_y
        resized_face = resized_face[0:clip_h, 0:clip_w]
        blurred_mask = blurred_mask[0:clip_h, 0:clip_w]

        roi = output_frame[paste_y:paste_y2, paste_x:paste_x2]
        alpha = blurred_mask.astype(float) / 255.0
        alpha = alpha[..., np.newaxis]

        blended = (resized_face.astype(float) * alpha + roi.astype(float) * (1 - alpha)).astype(np.uint8)
        output_frame[paste_y:paste_y2, paste_x:paste_x2] = blended

        if show_markers:
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            landmarks = predictor(gray, face)
            for n in range(68):
                lx = landmarks.part(n).x
                ly = landmarks.part(n).y
                cv2.circle(output_frame, (lx, ly), 2, (0, 255, 0), -1)

    cv2.imshow("Video - Enlarged Head", output_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        show_markers = not show_markers

cap.release()
cv2.destroyAllWindows()
