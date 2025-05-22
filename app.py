import gradio as gr
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Load default model
models = {
    "v1": "fracture_detection_v1_best.pt",
    "v2": "fracture_detection_v2_best.pt",
    "v3": "fracture_detection_v3_best.pt"
}

# Initial model
current_model = YOLO(models["v3"])

def predict(image, model_version):
    global current_model

    # Load selected model if different
    model_path = models[model_version]
    if current_model.model != model_path:
        current_model = YOLO(model_path)

    results = current_model.predict(image)[0]

    annotated_image = results.plot()

    # Count and list detected classes
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]
        detections.append(f"{label} ({conf:.2f})")

    summary = "No fractures detected." if not detections else f"Detected {len(detections)} fracture(s):\n" + "\n".join(detections)

    return annotated_image, summary

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload X-ray"),
        gr.Dropdown(choices=["v1", "v2", "v3"], value="v3", label="Model Version")
    ],
    outputs=[
        gr.Image(label="Annotated X-ray"),
        gr.Textbox(label="Detection Summary")
    ],
    title="Fracture Detection with YOLOv8",
    description="Upload an X-ray and select a YOLOv8 model version to detect bone fractures."
)

# Launch the interface locally
interface.launch()