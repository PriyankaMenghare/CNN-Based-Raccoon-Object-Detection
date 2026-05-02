# CNN-Based Raccoon Object Detection 🦝🔍

A deep learning object detection system that detects raccoons in images using a 
VGG16-based CNN with Selective Search region proposals (R-CNN approach). Trained 
on 200+ annotated raccoon images with bounding box annotations, achieving 99.5% 
validation accuracy.

---

## 🧩 Overview

This project implements a **Region-based CNN (R-CNN)** pipeline for raccoon object 
detection. It uses OpenCV's Selective Search to generate region proposals and a 
fine-tuned VGG16 model to classify each region as raccoon or background, with 
bounding boxes drawn around detected raccoons.

---

## 🔬 Methodology

### Pipeline

```
Input Image → Selective Search → Region Proposals → VGG16 Classifier → Bounding Box
```

### 1. Data Acquisition
- Dataset: 200+ raccoon images with bounding box annotations in CSV format
- Annotation format: `filename, width, height, class, xmin, ymin, xmax, ymax`
- Dataset cloned from GitHub repository

### 2. Region Proposal — Selective Search
- OpenCV's `ximgproc.segmentation.createSelectiveSearchSegmentation()`
- Fast Selective Search mode used to generate candidate regions
- IoU (Intersection over Union) used to label proposals as raccoon or background

### 3. IoU Calculation
```python
iou = intersection_area / (bb1_area + bb2_area - intersection_area)
```
- IoU ≥ threshold → raccoon (positive sample)
- IoU < threshold → background (negative sample)

### 4. Model — VGG16 (Transfer Learning)
- **Base:** VGG16 pretrained on ImageNet (138M parameters)
- **Input size:** 224×224×3
- **Custom head:** Flatten → Dense(4096) → Dense(4096) → Dense(2, softmax)
- **Fine-tuned** on raccoon region proposals

### 5. Training
| Parameter | Value |
|-----------|-------|
| Train samples | 5,754 |
| Test samples | 640 |
| Test split | 10% |
| Epochs | 5 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Callbacks | ModelCheckpoint, EarlyStopping (patience=100) |
| Augmentation | Horizontal flip, Vertical flip, Rotation (90°) |
| Accelerator | GPU (T4) |

---

## 📊 Results

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 1 | 97.41% | **99.37%** | 0.0178 ✅ |
| 2 | 98.28% | 98.59% | 0.0428 |
| 3 | 98.52% | 99.22% | 0.0191 |
| 4 | 98.73% | 98.44% | 0.0362 |
| 5 | 98.73% | **99.53%** | 0.0387 |

Best model saved at **Epoch 1** (lowest val_loss: 0.0178) as `ieeercnn_vgg16_1.keras`

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.10+
- TensorFlow / Keras
- OpenCV (with contrib modules for Selective Search)
- Google Colab with TPU (recommended)

### Install Dependencies
```bash
pip install tensorflow opencv-contrib-python pandas numpy matplotlib scikit-learn
```

### Run
```bash
# Clone the dataset
git clone https://github.com/PriyankaMenghare/CNN-Based-Raccoon-Object-Detection.git

# Open and run all cells
jupyter notebook CNN_Based_Racoon_Object_Detection.ipynb
```

---

## 📁 File Structure

```
CNN-Based-Raccoon-Object-Detection/
├── CNN_Based_Racoon_Object_Detection.ipynb   # Main notebook
├── train_labels.csv                          # Training bounding box annotations
├── test_labels.csv                           # Testing bounding box annotations
├── Images/                                   # Raccoon image dataset
│   ├── raccoon-1.jpg
│   ├── raccoon-2.jpg
│   └── ...
└── README.md
```

---

## 📝 License

MIT License — for academic use.
