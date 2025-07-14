import os
import sys
from ultralytics import YOLO

def run_yolo(task, source, model=None, show=False):
    if model is None:
        if task == "detect":
            model = "yolov8n.pt"
        elif task == "segment":
            model = "yolov8x-seg.pt"
        elif task == "classify":
            model = "yolov8x-cls.pt"
        else:
            raise ValueError("Task not supported")

    # Load the model
    yolo_model = YOLO(model)
    # Run prediction
    results = yolo_model.predict(source=source, show=show)
    return results

if __name__ == "__main__":
    input_file = "input.jpg"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    run_yolo("detect", input_file)
    run_yolo("segment", input_file)
    run_yolo("classify", input_file)