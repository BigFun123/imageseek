import cv2
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