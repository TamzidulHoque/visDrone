# Drone Human Detection & Counting System

## Overview
This project was developed for the ANTLINGS Internship Program Technical Assessment (AI/ML).

The goal of this project is to build a computer vision pipeline capable of:

- Detecting humans and cars from aerial/drone imagery
- Counting humans
- Visualizing detections with bounding boxes
- Performing optional object tracking

The project uses the VisDrone dataset and a YOLOv8-based object detection pipeline.

---

# Project Objectives

The system is designed to:

- Detect people and vehicles from drone footage
- Count the total number of humans detected in a frame
- Visualize predictions in images and videos
- Analyze challenges in aerial computer vision
- Demonstrate understanding of training, inference, and evaluation workflows

---

# Dataset

Dataset Used:
VisDrone Dataset

 - The dataset contains aerial images captured from drones under various real-world conditions.
## Dataset Characteristics

- Aerial/drone-view imagery
- Small object detection
- Dense traffic and crowd scenes
- Multiple object classes
- Real-world environmental variations
- Morning, noon, evening and night view
## Classes in the Dataset

The VisDrone dataset includes classes such as:

- pedestrian
- people
- bicycle
- car
- van
- truck
- bus
- motor
- awning-tricycle
- tricycle

Although the model was trained on all classes, the final application focuses mainly on:

- Human detection
- Car detection
- Human counting

---

# Dataset Structure

The original VisDrone dataset was provided with separate train, validation, and test directories containing independent image and label folders.

Dataset/
│
|__VisDrone2019-DET-test-challenge
|   |_images
|
├── VisDrone2019-DET-test-dev/
│   ├── images/
│   ├── labels/
│ 
│
├── VisDrone2019-DET-train/
│   ├── images/
│   ├── labels/
│   
│__VisDrone2019-DET-val
|   |__images/
|   |__labels/
|   |__labels.cache
|
└── data.yaml

For compatibility with the YOLOv8 training pipeline, the dataset was reorganized into a unified YOLO directory structure:

Dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml

Each label file contains:

class_id x_center y_center width height

The coordinates are normalized between 0 and 1.

---

# Dataset Understanding & Challenges

Several challenges were observed while working with the dataset:

## 1. Small Object Detection

Most humans and vehicles appear very small because the images are captured from drones flying at high altitude.

## 2. Dense Scenes

Many images contain crowded roads and pedestrian areas where multiple objects overlap.

## 3. Occlusion

Objects are partially hidden behind other objects.

## 4. Scale Variation

Objects appear at different sizes depending on drone altitude.

## 5. Domain Shift

The trained model may perform best on aerial/drone imagery but may struggle on close-range webcam images because the training distribution differs significantly due to lack of close range and COCO datasets.

---

# Preprocessing & Augmentation

The dataset was prepared in YOLO-compatible format. So, Choosing YOLOv8n for faster and industrial level model generation.

## Preprocessing Steps

- Image resizing (image_size: 640*640 given)
- YOLO label formatting (Found Formatted by default from downloads)
- Train/validation split (Found splitted by default from downloades)
- Data organization into image and label folders (YOLO automatically generated label.cache)

## Data Augmentation

YOLOv8 automatically applied several augmentations during training:

- Mosaic augmentation
- Horizontal flipping
- Scaling
- Translation
- HSV color augmentation
- Perspective transformation

These augmentations improved robustness against:

- Lighting variation
- Small object detection
- Drone movement
- Scale changes

---

# Model Training

## Model Used

- YOLOv8n from Ultralytics

The nano version was selected because it is lightweight and computationally efficient.

## Training Configuration

- Framework: Ultralytics YOLOv8
- Image Size: 640 (most suitable)
- Epochs: 50 (100 would be better but short time)
- Dataset: VisDrone
- Training Type: Fine-tuning
So pre-trained COCO dataset model is being further trained  

## Training Pipeline

The model was fine-tuned using pretrained YOLOv8 weights.

Example training code:

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    model.train(
        data="visdrone.yaml",
        epochs= 50,
        imgsz=640,
        batch=8,
        device= 0, 
        project= r"D:\Assessment_VisDrone\yolo_results",
    )
```

---

# Detection Pipeline

The detection system:

1. Loads the trained YOLOv8 model
2. Reads image/video frames
3. Runs inference
4. Draws bounding boxes
5. Displays object classes
6. Counts humans
7. Visualizes final outputs

---

# Human Counting Logic

Human counting was implemented using simple detection-based counting.

Logic:

- If detected class == person/pedestrian/people
- Increase counter by 1

This approach works effectively for frame-level counting.

---

# Object Tracking
I sequentially used all of these to compare which is better:
 - DeepSort (Primarily selected)
 - ByteSort
 - BotSort
 - cv2 frames in loop

## DeepSort is better because:
 - Bounding box looks sharp
 - Can Distinguish the densed people more accurately 
 - YOLO detects objects
 - DeepSORT assigns track IDs
 - Objects are tracked frame-by-frame

---

# Results

The system successfully:

- Detected humans and van
- Counted humans
- Generated thinner bounding-box with better visualizations
- Performed object tracking on video sequences

# Observations & Analysis

The model performed best on:
- Clear videos
- Marginal distance objects

# But limited to:
- Much far image visuals and blurry videos is bigger challenge
- Any similar shaped undesired object is faltly detected
- Some objects are still not detected. So more training needed (100 epochs, 16 batches, 1280 img_sizes).
- Struggles to detect very closer objects while in webcam.

---

## Key Technical Insight

The primary limitation was domain mismatch.

The model was trained mainly on aerial drone imagery, therefore it generalizes best to drone-view scenes and less effectively to frontal or close-range perspectives.

---

# Technologies Used

    - Python
    - Ultralytics YOLOv8
    - OpenCV
    - DeepSORT
    - NumPy
    - matplotlib
Aditionally:
    - CUDA (GPU use)

---

# Sample Features
1 image and 5 videos with different trackers used
- Human Detection
- Car Detection
- Human Counting
- Bounding Box Visualization
- Video Processing
- Object Tracking

---

# Future Improvements

Possible future improvements include:

- Training with larger YOLO models (YOLOv8s/m)
- Combining VisDrone with COCO dataset
- Better counting algorithms
- Improved multi-object tracking
- GPU-based faster training
- Real-time drone deployment

---

# How to Run

## Install Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install deep-sort-realtime
```

## Run Detection

```bash
python src/image_detection.py
```
IMPORTANT: A newer image path have to be given at the img_path as input
---

# Conclusion

This project demonstrates a complete aerial computer vision workflow including:

- Dataset understanding
- Preprocessing
- Model training
- Inference
- Visualization
- Counting
- Tracking
- Result analysis

The implementation highlights practical challenges of drone-based object detection systems and demonstrates understanding of real-world AI/ML engineering workflows.

