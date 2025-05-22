# Fracture Detection UI with YOLOv8

This project provides an interactive, locally deployable user interface for detecting bone fractures in X-ray images using a YOLOv8 object detection model. Built with Gradio and Python, the interface is simple to run, requires no web development knowledge, and is suitable for both research and clinical prototyping.

## Features

- Upload and analyze individual X-ray images
- Automatically detects and labels fractures (e.g., humerus, ulna, femur)
- Displays an annotated image with bounding boxes
- Provides a text-based summary of the number of detections and their confidence scores
- Model version selector (e.g., v1, v2, v3)
- Runs fully offline in a local browser environment

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/betulsit/fracture-detection-ui.git
cd fracture-detection-ui
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your trained YOLOv8 model

Model files (`.pt`) are not included in the repository due to size limitations. You must manually add them to the project folder.

The UI includes a dropdown menu to select between different trained models:
- `v1` → fracture_detection_v1_best.pt
- `v2` → fracture_detection_v2_best.pt
- `v3` → fracture_detection_v3_best.pt (default)

To add additional model versions:
- Save the new `.pt` file in the project folder.
- Update the `models` dictionary in `app.py` accordingly.

## Running the Application

To start the local web interface:

```bash
python app.py
```

Once running, the interface will be available at:

```
http://127.0.0.1:7860
```

## Project Structure

```
fracture-detection-ui/
├── app.py                              # Main UI code
├── requirements.txt                    # Dependencies
├── .gitignore                          # Git ignore rules
├── README.md                           # This file
├── fracture_detection_v1_best.pt       # YOLOv8 model v1 (manual)
├── fracture_detection_v2_best.pt       # YOLOv8 model v2 (manual)
├── fracture_detection_v3_best.pt       # YOLOv8 model v3 (default, manual)
└── venv/                               # Virtual environment (not tracked)
```