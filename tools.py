import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

corner_length = 5  # Length of the corner lines

def drawOutlineBox(frame, bounds, color=(0, 255, 0), thickness=2):
    # Top-left corner
    x1, y1, x2, y2 = bounds
    
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
    
def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def draw_unicode_text(frame, text, position, font_path="arial.ttf", font_size=14, color=(0, 10, 0)):
    # Convert frame to PIL image
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)