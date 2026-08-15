# GeoGuessr Hackathon - Geolocation Prediction

This repository contains our team's solution and models for the **GeoGuessr Hackathon / Geolocation Prediction Challenge**. The goal is to predict the latitude, longitude, and an uncertainty radius (in kilometers) for a given street view / outdoor image.

## Repository Structure

- [train.py](file:///train.py): Script to train the multi-stage geolocation prediction model.
- [dataset.py](file:///dataset.py): PyTorch dataset definitions, data loading, and preprocessing (including image transformations).
- [model.py](file:///model.py): Deep learning model architecture for feature extraction and coordinate regression.
- [inference.py](file:///inference.py): Evaluates trained checkpoints on the test set and generates the `submission.csv`.
- [streetclip_zeroshot.py](file:///streetclip_zeroshot.py): Alternative zero-shot country classification baseline using [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) (text-vision) to weight country centroid coordinates and dynamically adjust target search radii.
- [run.py](file:///run.py): Pipeline script that automatically runs validation, training, and inference consecutively.
- [analyze_submission.py](file:///analyze_submission.py): Script to analyze coordinates, distribution, and radius statistics of the generated submission.
- [Geoguessr_hackathon.pdf](file:///Geoguessr_hackathon.pdf): Challenge document/guidelines.
- [.gitignore](file:///.gitignore): Configured to ignore raw datasets, large wheels, zip files, and PyTorch checkpoints (`.pt`, `.pth`).

---

## Setup & Requirements

### Dependencies
Install the required packages. A CUDA-enabled environment is highly recommended for training.
```bash
pip install torch torchvision transformers pillow numpy pandas geojson
```

### Dataset Structure
The dataset directory should be structured as follows:
```text
geolocation-prediction/
├── country_boundaries.geojson
├── sample_submission.csv
├── training_dataset/
│   └── noised_dataset/
│       └── images/       # Training images (.jpg)
└── test_images_sampled/  # Test images (.jpg)
```

---

## How to Run

### 1. End-to-End Deep Learning Pipeline
You can run the training and inference pipeline using `run.py`:
```bash
python run.py
```
This script will:
1. Verify directories, file structures, and GPU availability.
2. Train the model using the parameters defined in `train.py` (Stage 1, Stage 2, and Stage 3 fine-tuning).
3. Run inference on the test set using the best checkpoints, producing `submission.csv`.

### 2. Zero-Shot StreetCLIP Baseline
Alternatively, you can run the zero-shot StreetCLIP classifier:
```bash
python streetclip_zeroshot.py
```
This uses the CLIP model pre-trained on GeoGuessr street views to match the image against country name text prompts (e.g. *"A street view photo in France"*), computes similarity probabilities, weights the top 5 country coordinates, and outputs a `submission.csv` with confidence-based radii.

---

## Submission Output Format
The final predictions are stored in `submission.csv` containing:
- `image_id`: Unique filename of the image.
- `pred_lat`: Predicted latitude.
- `pred_lon`: Predicted longitude.
- `pred_radius_km`: Predicted uncertainty radius (search circle size).
