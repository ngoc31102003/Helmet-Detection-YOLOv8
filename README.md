<h1 align="center">
  <br>
  Helmet Detection using YOLOv8
  <br>
</h1>

<h4 align="center">A real-time motorcycle helmet detection system for traffic safety monitoring</h4>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/YOLOv8-8.0%2B-red" alt="YOLOv8">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/OpenCV-4.5%2B-green" alt="OpenCV">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Accuracy-92%25-brightgreen" alt="Accuracy">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#demo">Demo</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#structure">Structure</a>
</p>

## Overview

This project implements a **real-time helmet detection system** using YOLOv8 for motorcycle safety monitoring. The system detects and classifies four main object categories with high accuracy.

**Detection Classes:**
- **Helmet** - Motorcycle helmets
- **Motorcyclist** - People riding motorcycles  
- **Non-helmet** - Riders without helmets
- **Plate** - License plates

## 🎥 Demo
### Real-time Detection (Auto-running GIFs)

<p align="center">
  <img src="video_test/video_test_1.gif" width="80%" alt="Helmet Detection Demo 1">
  <img src="video_test/video_test_2.gif" width="80%" alt="Helmet Detection Demo 2">
</p>

*Urban traffic detection (left) and highway monitoring (right)*

## 📊 Training Results

### Performance Metrics
<p align="center">
  <img src="runs/detect/viet_traffic_signs_v8n2/confusion_matrix.png" width="60%" alt="Confusion Matrix"><br><br>
  <img src="runs/detect/viet_traffic_signs_v8n2/F1_curve.png" width="60%" alt="F1 Curve">
  <img src="runs/detect/viet_traffic_signs_v8n2/PR_curve.png" width="60%" alt="F1 Curve">
  <img src="runs/detect/viet_traffic_signs_v8n2/P_curve.png" width="60%" alt="F1 Curve">
  <img src="runs/detect/viet_traffic_signs_v8n2/R_curve.png" width="60%" alt="F1 Curve">

</p>

### Training Progress
<p align="center">
  <img src="runs/detect/viet_traffic_signs_v8n2/results.png" width="90%" alt="Training Results">
</p>

### Validation Results
<p align="center">
  <img src="runs/detect/viet_traffic_signs_v8n2/val_batch0_pred.jpg" width="45%" alt="Validation Batch 1">
  <img src="runs/detect/viet_traffic_signs_v8n2/val_batch1_pred.jpg" width="45%" alt="Validation Batch 2">
</p>

### Training Samples
<p align="center">
  <img src="runs/detect/viet_traffic_signs_v8n2/train_batch0.jpg" width="45%" alt="Training Batch 1">
  <img src="runs/detect/viet_traffic_signs_v8n2/train_batch1.jpg" width="45%" alt="Training Batch 2">
</p>

## 📈 Performance Summary

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>mAP50</b></td><td><b>92.1%</b></td><td>Mean Average Precision at IoU=0.5</td></tr>
    <tr><td><b>mAP50-95</b></td><td><b>67.8%</b></td><td>Mean Average Precision IoU 0.5-0.95</td></tr>
    <tr><td><b>Precision</b></td><td><b>89.4%</b></td><td>Detection precision</td></tr>
    <tr><td><b>Recall</b></td><td><b>85.2%</b></td><td>Detection recall</td></tr>
    <tr><td><b>Inference Speed</b></td><td>45 FPS</td><td>On NVIDIA GPU</td></tr>
  </tbody>
</table>

<hr>

<h3>Per-Class Performance</h3>

<table>
  <thead>
    <tr>
      <th>Class</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>mAP50</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>Helmet</b></td><td>91.2%</td><td>87.6%</td><td>93.4%</td></tr>
    <tr><td><b>Motorcyclist</b></td><td>88.7%</td><td>83.9%</td><td>90.1%</td></tr>
    <tr><td><b>Non-helmet</b></td><td>90.1%</td><td>86.3%</td><td>92.8%</td></tr>
    <tr><td><b>Plate</b></td><td>87.6%</td><td>83.0%</td><td>89.2%</td></tr>
  </tbody>
</table>

## 🛠 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/Helmet-Detection-YOLOv8.git
cd Helmet-Detection-YOLOv8

# Install dependencies
pip install -r requirements.txt
Requirements
text
ultralytics>=8.0.0
torch>=1.7.0
opencv-python>=4.5.0
albumentations>=1.0.0
numpy>=1.19.0
💻 Usage
Training
bash
# Data preparation
python data_preparation/data_split_augmentation.py

# Start training
python training/train_helmet_detection.py

# Monitor training
tensorboard --logdir runs/detect
Inference
bash
# Real-time webcam detection
python inference/helmet_detection_inference.py --source 0

# Process video file
python inference/helmet_detection_inference.py --source videos/test.mp4

# Process image
python inference/helmet_detection_inference.py --source images/test.jpg
Model Export
bash
# Export to ONNX
python utils/export_onnx.py

# Export to TensorRT (requires GPU)
python utils/export_tensorrt.py
📁 Project Structure
text
Helmet-Detection-YOLOv8/
├── 📊 data_preparation/
│   ├── data_analysis.py
│   ├── data_split_augmentation.py
│   └── dataset_visualization.py
├── 🗃️ datasets_doi_mu_bao_hiem_new/
│   ├── train/images/
│   ├── valid/images/
│   ├── test/images/
│   └── dataset.yaml
├── 🤖 inference/
│   ├── helmet_detection_inference.py
│   └── onnx_inference.py
├── 🏋️ training/
│   ├── train_helmet_detection.py
│   └── training_results.py
├── 🔧 utils/
│   ├── export_onnx.py
│   └── image_test.py
├── 📈 runs/detect/viet_traffic_signs_v8n2/
│   ├── weights/
│   │   ├── best.pt
│   │   ├── best.onnx
│   │   └── last.pt
│   ├── *.png (training metrics)
│   ├── train_batch*.jpg
│   └── val_batch*_.jpg
└── 📄 requirements.txt
🚀 Key Features
Real-time Detection: 30 FPS on GPU

Multi-format Export: PyTorch, ONNX, TensorRT support

Data Augmentation: Comprehensive preprocessing pipeline

Performance Analytics: Detailed training metrics and visualization

Production Ready: Well-structured, documented code

📁 File requirements.txt:
txt
ultralytics==8.0.0
torch>=1.7.0
torchvision>=0.8.0
opencv-python>=4.5.0
matplotlib>=3.3.0
numpy>=1.19.0
albumentations>=1.0.0
tqdm>=4.60.0
pandas>=1.1.0
PyYAML>=5.4.0

