import gradio as gr
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path

# --- Model Paths and Constants ---
# NOTE: Update model paths as needed.
models = {
    "v1": "models/fracture_detection_v1_best.pt",
    "v3": "models/fracture_detection_v3_best.pt",
    "v2_base": "models/fracture_detection_v2_best.pt",
    "v2_masked": "models/fracture_detection_v2_masked_best.pt",
    "v2_masked (Direct)": "models/fracture_detection_v2_masked_best.pt"
}

MASKING_RULES = {
    # NOTE: These class IDs must match the output of the 'v2_base' model.
    0: 0.15, 1: 0.2, 2: 0.15, 3: 0.15, 4: 0.3,
    5: 0.2, 6: 0.3, 7: 0.2, 8: 0.3, 9: 0.15,
    10: 0.2, 11: 0.2, 12: 0.2, 13: 0.2,
}

# --- Model Pre-loading ---
print("Loading frequently used YOLO models...")
try:
    loaded_yolo_models = {
        "v2_base": YOLO(models["v2_base"]),
        "v2_masked": YOLO(models["v2_masked"]),
    }
    print("v2_base and v2_masked models loaded.")
except Exception as e:
    print(f"ERROR: Initial model loading failed: {e}")

# --- Helper Functions ---

def apply_pseudo_mask_from_detections(original_image_np, boxes_data, image_width, image_height):
    """Generates a pseudo-mask from pre-detection results."""
    pseudo_mask_img = np.zeros_like(original_image_np)
    mask_applied = False
    
    # --- DEBUGGING: Lower the confidence threshold temporarily to see if anything gets through ---
    CONFIDENCE_THRESHOLD = 0.1 # Temporarily lowered from 0.25 to 0.1 for testing

    print(f"  - Applying mask with confidence threshold: {CONFIDENCE_THRESHOLD}")
    if boxes_data is not None and len(boxes_data) > 0:
        for box in boxes_data:
            confidence = float(box.conf[0])
            print(f"    > Checking box with confidence: {confidence:.2f}") # DEBUG PRINT

            if confidence < CONFIDENCE_THRESHOLD:
                continue # Skip boxes below the threshold

            mask_applied = True # If at least one box passes, we will apply a mask
            
            # Continue with the rest of the logic for the passed box...
            if box.xywhn.nelement() == 0: continue
            cx_norm, cy_norm, bw_norm, bh_norm = box.xywhn[0].cpu().numpy()
            cx, cy, bw, bh = cx_norm * image_width, cy_norm * image_height, bw_norm * image_width, bh_norm * image_height
            padding_factor = MASKING_RULES.get(int(box.cls[0]), 0.2)
            pad_w, pad_h = bw * padding_factor, bh * padding_factor
            x1 = max(int(cx - bw / 2 - pad_w), 0)
            y1 = max(int(cy - bh / 2 - pad_h), 0)
            x2 = min(int(cx + bw / 2 + pad_w), image_width - 1)
            y2 = min(int(cy + bh / 2 + pad_h), image_height - 1)
            if x1 < x2 and y1 < y2:
                pseudo_mask_img[y1:y2, x1:x2] = original_image_np[y1:y2, x1:x2]

    if not mask_applied:
        print("  - DEBUG: No boxes passed the confidence threshold. Masking failed.")
        return None, False

    return pseudo_mask_img, True

# --- Main Prediction Function ---

def predict(image_np, model_version_key):
    (single_output, pseudo_mask_output, final_masked_output, summary) = (None, None, None, "Processing not started.")

    try:
        if model_version_key == "v2_masked":
            print("\n--- INITIATING TWO-STAGE ANALYSIS ---")

            # Stage 1: Pre-detection
            base_model = loaded_yolo_models.get("v2_base")
            print("1. Running pre-detection with v2_base...")
            base_results = base_model.predict(image_np, verbose=False)[0]
            print(f"   -> Found {len(base_results.boxes) if base_results.boxes else 0} total boxes (before filtering).")
            h, w = base_results.orig_shape

            # Stage 2: Pseudo-Mask Generation
            print("2. Generating pseudo-mask from detections...")
            generated_pseudo_mask, mask_applied = apply_pseudo_mask_from_detections(
                base_results.orig_img, base_results.boxes, w, h
            )

            if not mask_applied:
                summary = "Mask generation failed (no detections from v2_base passed the confidence threshold)."
                print(f"--- PROCESS HALTED: {summary} ---")
                # To help debug, show what the v2_base model found, even if it's low confidence
                # This will appear in the single-stage output area if you switch the model selector
                single_output_for_debug = base_results.plot()
                return single_output_for_debug, None, None, summary

            pseudo_mask_output = generated_pseudo_mask.copy()
            print("   -> Pseudo-mask generated successfully.")

            # Stage 3: Detailed Detection
            masked_model = loaded_yolo_models.get("v2_masked")
            print("3. Running detailed detection with v2_masked...")
            masked_results = masked_model.predict(generated_pseudo_mask, verbose=False)[0]
            final_masked_output = masked_results.plot()
            print(f"   -> Found {len(masked_results.boxes) if masked_results.boxes else 0} boxes with v2_masked.")


            # Prepare summary for final results
            final_detections = []
            if masked_results.boxes is not None:
                for box in masked_results.boxes:
                    label = masked_model.names[int(box.cls[0])]
                    final_detections.append(f"{label} ({float(box.conf[0]):.2f})")
            summary = "No fractures detected by v2_masked." if not final_detections else f"Detected {len(final_detections)} fracture(s) with v2_masked:\n" + "\n".join(final_detections)
            print("--- ANALYSIS COMPLETE ---")

        else:
            # --- STANDARD SINGLE-STAGE DETECTION ---
            print(f"\n--- INITIATING SINGLE-STAGE ANALYSIS with model: {model_version_key} ---")
            model_path = models[model_version_key]
            active_model = loaded_yolo_models.get(model_version_key)
            if not active_model or active_model.ckpt_path != model_path:
                 active_model = YOLO(model_path)
            results = active_model.predict(image_np, verbose=False)[0]
            single_output = results.plot()
            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    label = active_model.names[int(box.cls[0])]
                    detections.append(f"{label} ({float(box.conf[0]):.2f})")
            summary = "No fractures detected." if not detections else f"Detected {len(detections)} fracture(s):\n" + "\n".join(detections)
            summary += f" (Model: {model_version_key})"
            print("--- ANALYSIS COMPLETE ---")

    except Exception as e:
        summary = f"An ERROR occurred: {e}"
        print(summary)

    return single_output, pseudo_mask_output, final_masked_output, summary


# --- Gradio UI Definition ---

model_choices = list(models.keys())
default_model_choice = "v2_masked"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# YOLOv8 Fracture Detection")
    gr.Markdown(
        """
        Upload an X-ray to detect bone fractures.
        - **Standard models (v1, v3, v2_base):** Process the uploaded image directly.
        - **v2_masked:** Performs a two-stage analysis on unmasked images.
        - **v2_masked (Direct):** Processes an image you have already masked yourself.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload X-ray Image")
            model_selector = gr.Dropdown(choices=model_choices, value=default_model_choice, label="Select Model Version")
            submit_btn = gr.Button("Detect Fractures", variant="primary")
        with gr.Column(scale=2):
            with gr.Column(visible=True) as single_stage_output_area:
                single_stage_output_image = gr.Image(label="Annotated Result")
            with gr.Row(visible=False) as two_stage_output_area:
                pseudo_mask_output_image = gr.Image(label="Generated Pseudo-Mask")
                final_masked_output_image = gr.Image(label="Final Result (on Mask)")
            summary_output = gr.Textbox(label="Detection Summary", lines=10, interactive=False)

    def update_ui_visibility(model_version_key):
        if model_version_key == "v2_masked":
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)

    model_selector.change(
        fn=update_ui_visibility,
        inputs=model_selector,
        outputs=[single_stage_output_area, two_stage_output_area]
    )

    submit_btn.click(
        fn=predict,
        inputs=[image_input, model_selector],
        outputs=[single_stage_output_image, pseudo_mask_output_image, final_masked_output_image, summary_output]
    )


if __name__ == '__main__':
    print("Launching Gradio interface... Access at http://127.0.0.1:7860")
    demo.launch()