import time

class TrackedFace:
    def __init__(self, box, name, score):
        self.box = box  # last bounding box
        self.name = name
        self.score = score
        self.last_seen = time.time()

    def iou(self, new_box):
        # Simple Intersection over Union to match old and new boxes
        x1, y1, x2, y2 = self.box
        a1, b1, a2, b2 = new_box
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box_area = (x2 - x1) * (y2 - y1)
        new_area = (a2 - a1) * (b2 - b1)
        union = box_area + new_area - inter_area

        return inter_area / union if union != 0 else 0