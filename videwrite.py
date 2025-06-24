import cv2
from datetime import datetime
import os

def setup(fps, width, height):
    # Timestamped output
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_folder = f"detections/{timestamp_str}"
    os.makedirs(output_folder, exist_ok=True)
    #fps = 30  # Frames per second for the video
    #width, height = 640, 480  # Video dimensions

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out_path = os.path.join(output_folder, f"{timestamp_str}.mp4")
    return cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))