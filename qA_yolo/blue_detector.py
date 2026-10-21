"""Blue inference uses the same YOLO model, filtered to B-prefixed classes."""
from yolo_detector import detect as infer

def detect_blue(frame, **kwargs):
    return infer(frame, color='blue', **kwargs)
