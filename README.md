# Door Detection for Construction Drawings

A PyTorch-based computer vision system for detecting doors in architectural and construction drawings using YOLOv8 with domain adaptation.

## Overview

This project demonstrates a robust door detection system that bridges the domain gap between clean architectural drawings and noisy construction documents. The system achieves 72.8% mAP through systematic domain adaptation and intelligent preprocessing.

### Key Features
- **Multi-domain training**: 981 labeled images across architectural and construction domains
- **Transfer learning pipeline**: Hand-labeled 371 doors from sample_drawings.pdf enabling domain adaptation from clean architectural drawings (18% → 69% recall)
- **Preprocessing proof of concept**: Connected component analysis for floor plan extraction from visually noisy construction documents 


## Results Summary

| Metric | Performance | Notes |
|--------|-------------|-------|
| mAP@0.5 | 72.8% | Strong generalization capability across diverse architectural drawing styles |
| Precision | 81.5% | Low false positive rate |
| Recall | 69.0% | Comprehensive door detection |


## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup
```bash
# Clone or extract the project
cd door_detection_project

# Install dependencies
pip install -r requirements.txt
```

### Run scripts
```bash
# run training 
python train_door_detector.py

# run preprocessing PoC
python construction_doc_cropper.py --image="image_path"

# run filter for Floor Plans 500 dataset (optional), assumes you have dataset at top level dir
python door_plans_2_floor_plans.py


# run inference 
yolo detect predict model="./models/best_door_model.pt" source="image_path"
```
