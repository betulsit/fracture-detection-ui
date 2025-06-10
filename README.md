# Fracture Detection UI with YOLOv8

This project provides an interactive, locally deployable user interface for detecting bone fractures in X-ray images using YOLOv8 models. Built with Gradio and Python, the interface allows for simple, single-stage fracture detection as well as an advanced two-stage analysis pipeline for specialized models.

## Features

### Multiple Analysis Modes

* **Standard Detection**: Upload an unmasked X-ray and get immediate, annotated results.
* **Two-Stage Analysis**: For specialized models (`v2_masked`), performs an automated pre-detection, generates a "pseudo-mask" to focus on relevant areas, and then runs a detailed analysis.
* **Direct Masked Detection**: Allows users to upload their own pre-masked images for analysis with compatible models.

### Dynamic User Interface

* The UI layout intelligently adapts, showing single or multiple output images depending on the selected analysis mode.

### Comprehensive Results

* Displays annotated images with bounding boxes, class labels, and confidence scores.

### Model Selection

* A dropdown menu allows for easy switching between different model versions and analysis modes.

### Local & Offline

* Runs fully on your local machine in a web browser, ensuring data privacy.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/betulsit/fracture-detection-ui.git
cd fracture-detection-ui
```

### 2. Set up a virtual environment

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

The trained model files (`.pt`) are hosted on Google Drive.

Download the models from the following shared folder:

https://drive.google.com/drive/folders/1bABHMNOaeEN5aYWxqEZYc_zB5bQaZRvl?usp=sharing 

Create a `models` folder inside your `fracture-detection-ui` project directory.

Place the downloaded `.pt` files inside this new `models` folder.

Update model paths in `app.py`. Make sure the paths in the `models` dictionary point to the new folder. For example:

```python
# app.py
models = {
    "v1": "models/fracture_detection_v1_best.pt",
    "v3": "models/fracture_detection_v3_best.pt",
    "v2_base": "models/fracture_detection_v2_best.pt",
    "v2_masked": "models/fracture_detection_v2_masked_best.pt",
    "v2_masked (Direct)": "models/fracture_detection_v2_masked_best.pt"
}
```

## Running the Application

To start the local web interface:

```bash
python app.py
```

Once running, the interface will be available at: [http://127.0.0.1:7860](http://127.0.0.1:7860)

## How to Use the Analysis Modes

### Standard Detection (`v1`, `v3`, `v2_base`)

1. Upload a standard, unmasked X-ray image.
2. Select `v1`, `v3`, or `v2_base`.
3. Click **Detect Fractures**.
4. **Result**: A single output image with bounding box annotations will be displayed.

### Two-Stage Analysis (`v2_masked`)

1. Upload a standard, unmasked X-ray image.
2. Select `v2_masked`.
3. Click **Detect Fractures**.
4. **Result**: Two output images will be displayed: the generated pseudo-mask and the final annotated result on that mask.

### Direct Masked Detection (`v2_masked (Direct)`)

1. Upload a pre-masked X-ray image.
2. Select `v2_masked (Direct)`.
3. Click **Detect Fractures**.
4. **Result**: A single output image showing the final detections on your provided masked image.

## Project Structure (Local)

```
fracture-detection-ui/
├── app.py                                          # Main Gradio UI application code
├── requirements.txt                                # Project dependencies
├── .gitignore                                      # Git ignore rules
├── README.md                                       # This file
├── models/                                         # Folder for trained models
│   ├── fracture_detection_v1_best.pt
│   ├── fracture_detection_v3_best.pt
│   ├── fracture_detection_v2_best.pt
│   └── fracture_detection_v2_masked_best.pt
└── venv/                                           # Virtual environment
```
