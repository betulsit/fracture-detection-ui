# Fracture Detection UI with YOLOv8

This project provides an interactive, locally deployable user interface for detecting bone fractures in X-ray images using a YOLOv8 object detection model. Built with Gradio and Python, the interface is simple to run, requires no web development knowledge, and is suitable for both research and clinical prototyping.

## Features

- Upload and analyze individual X-ray images
- Automatically detects and labels fractures (e.g., humerus, ulna, femur)
- Displays an annotated image with bounding boxes
- Provides a text-based summary of the number of detections and their confidence scores
- Runs fully offline in a local browser environment

## Requirements

The project requires the following Python packages:

- ultralytics
- gradio
- opencv-python

These are listed in the `requirements.txt` file.

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

Place your trained `best.pt` model file in the project root directory. This file is not tracked in the repository and should be added manually.

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
├── app.py              # Gradio interface code
├── best.pt             # YOLOv8 model file (user-provided)
├── requirements.txt    # Dependency list
├── .gitignore          # Git exclusions
└── venv/               # Virtual environment (excluded from repo)
```