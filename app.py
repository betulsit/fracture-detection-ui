import gradio as gr
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("best.pt")

# Inference function
def detect_fracture(image):
    # Run prediction
    results = model.predict(image, conf=0.25)
    boxes = results[0].boxes

    # Prepare annotated image
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Prepare detection summary
    if boxes is None or len(boxes.cls) == 0:
        summary = "No fractures detected."
    else:
        class_names = model.names
        summary_lines = [f"Detected {len(boxes.cls)} fracture(s):"]
        for cls_id, score in zip(boxes.cls, boxes.conf):
            class_name = class_names[int(cls_id)]
            summary_lines.append(f"- {class_name} ({score:.2f})")
        summary = "\n".join(summary_lines)

    return annotated, summary

# Gradio interface
interface = gr.Interface(
    fn=detect_fracture,
    inputs=gr.Image(type="numpy", label="Upload X-ray"),
    outputs=[
        gr.Image(type="numpy", label="Detected Fractures"),
        gr.Textbox(label="Detection Summary")
    ],
    title="Fracture Detection Demo",
    description="Upload an X-ray image to detect fractures using a YOLOv8 model."
)

# Launch the interface locally
interface.launch()