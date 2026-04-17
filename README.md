# Crop Disease Classification

Deep learning project for multi-class classification of crop diseases using leaf images. Built as a capstone project under the NgaoLabs Data Science Training Program.

---

## Executive Summary

| Attribute | Specification |
|-----------|-------------|
| **Problem Type** | Multi-class Image Classification (10 classes) |
| **Input Data** | Leaf images (RGB) |
| **Models** | Custom CNN, MobileNetV2, EfficientNetB0 |
| **Primary Metric** | Weighted F1-score (class imbalance sensitivity) |
| **Deployment** | Streamlit web application |

---

## Results & Insights

- ✅ **Transfer learning outperformed custom CNN** due to pre-trained ImageNet feature extraction
- ⚠️ **Generalization challenges** observed from controlled dataset conditions (lab imagery)
- 📊 **Model interpretability** via confidence visualization and top-5 prediction charts
- 🚀 **MobileNetV2 selected** as optimal accuracy-efficiency trade-off (88-93% validation accuracy)

---

## Problem Context

Crop diseases reduce agricultural productivity and are often diagnosed late due to limited access to expertise. In Kenya, **94% of farmers** are affected by pests and diseases, with extension officer-to-farmer ratios exceeding **1:1,500**.

Manual inspection is:
- ❌ Slow and inconsistent
- ❌ Expertise-dependent
- ❌ Reactive (2-3 week detection delays)

**Solution:** Automated image-based classification system for faster, scalable diagnosis.

---

## Approach

### Data Preparation

| Step | Operation | Parameters |
|------|-----------|------------|
| Resizing | Bicubic interpolation | 224×224 pixels |
| Normalization | Pixel scaling | [0, 1] range |
| Splitting | Stratified | 70/15/15 (train/val/test) |
| Augmentation | Rotation, shift, flip, zoom, brightness | Training set only |

**Augmentation Config:**
- Rotation: ±20°
- Width/Height shift: ±20%
- Horizontal flip: 50% probability
- Zoom: [0.8, 1.2]
- Brightness: [0.8, 1.2]

### Exploratory Data Analysis

- Class distribution assessment (imbalance ratio: 1.4:1)
- Sample visualization per class
- Image quality verification (dimensions, corruption detection)
- Duplicate detection (MD5 hash-based)

### Feature Engineering

- **Implicit feature extraction** via CNN architectures
- No manual feature engineering required
- Pre-trained ImageNet weights for transfer learning models

---

## Modeling

### Models Evaluated

| Model | Type | Parameters | Expected Accuracy |
|-------|------|------------|-------------------|
| Custom CNN | Baseline | ~1.2M | 75-85% |
| MobileNetV2 | Transfer Learning | 3.5M (~0.5M trainable) | 88-93% |
| EfficientNetB0 | Transfer Learning | 5.3M (~0.6M trainable) | 90-95% |

### Selection Rationale

- **Custom CNN:** Establishes baseline performance
- **Transfer Learning:** Leverages ImageNet features for improved generalization and training efficiency
- **MobileNetV2 Selected:** Optimal balance of accuracy, speed, and model size for deployment

### Training Strategy

| Parameter | Setting |
|-----------|---------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Initial Learning Rate | 0.0001 (frozen), 0.00001 (fine-tuning) |
| Epochs | 30 (10 frozen + 20 fine-tuning) |
| Batch Size | 32 |
| Early Stopping | Patience=5, monitor=val_loss |
| LR Reduction | ReduceLROnPlateau (factor=0.2, patience=3) |
| Class Weights | Computed from training distribution |

---

## Evaluation

### Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >90% | 91.2% |
| Precision (macro) | >85% | 89.4% |
| Recall (macro) | >85% | 88.7% |
| F1-score (macro) | >85% | 89.0% |
| **F1-score (weighted)** | **>87%** | **90.8%** |
| Cohen's Kappa | >0.80 | 0.84 |

### Validation

- Hold-out test set (15% of data)
- Stratified sampling to preserve class distribution
- Confusion matrix analysis per class

---

## Dataset

| Attribute | Specification |
|-----------|-------------|
| **Source** | [Kaggle – Five Crop Diseases Dataset](https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset) |
| **Original Size** | 13,324 images |
| **Original Classes** | 17 (5 crops) |
| **Filtered Classes** | **10** (3 crops: Corn, Potato, Wheat) |
| **Format** | RGB JPEG |
| **License** | CC0: Public Domain |

### Class Distribution

| Crop | Classes | Images |
|------|---------|--------|
| Corn | Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy | 4,252 |
| Potato | Early Blight, Late Blight, Healthy | 2,152 |
| Wheat | Brown Rust, Yellow Rust, Healthy | 2,942 |

*Note: Rice and Sugarcane excluded per project scope (Phase 2 expansion)*

---

## Deployment

**Platform:** Streamlit

### Features

| Feature | Description |
|---------|-------------|
| Image Upload | Drag-and-drop or file selector (JPG, PNG, JPEG) |
| Prediction Display | Predicted class + confidence percentage |
| Confidence Visualization | Top-5 predictions with bar chart |
| Advisory System | Rule-based treatment and prevention recommendations |
| History Log | Session-based scan tracking |
| Theme Toggle | Dark/Light mode |

### Live Demo

```
streamlit run app.py
```

Access at `http://localhost:8501`
Hosted at  `https://cropdiseasespredictioncnn-d2sxrpgrmhfgxbcotwiczz.streamlit.app/`
---

## Project Structure

```
crop_disease_prediction_CNN/
│
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/
│   └── deployment/             # Production models
│       ├── NeuralNest_MobileNetV2.keras
│       ├── class_names.json
│       ├── advisory_rules.json
│       └── label_encoder.pkl
│
├── data/
│   ├── processed/              # Cleaned datasets
│   │   ├── train_manifest.csv
│   │   ├── val_manifest.csv
│   │   ├── test_manifest.csv
│   │   └── metadata.json
│   │
│   └── splits/                 # Organized images
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Data preprocessing
│   ├── 03_model_training.ipynb# Model training & evaluation
│   └── 04_deployment.ipynb    # Deployment preparation
│
├── src/
│   ├── preprocessing.py       # Image preprocessing utilities
│   ├── model.py               # Model architectures
│   ├── train.py               # Training script
│   └── utils.py               # Helper functions
│
└── assets/
    └── logo.png               # Project logo
```

---

## Reproducibility

### Setup

```bash
# 1. Clone the repository
git clone git@github.com:karanja-dave/crop_disease_prediction_CNN.git

# 2. Navigate to project folder
cd crop_disease_prediction_CNN

# 3. Create virtual environment
python -m venv .venv

# 4. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run application
streamlit run app.py
```

### Requirements

```
streamlit>=1.28.0
tensorflow==2.21.0
pillow>=9.0.0
numpy>=1.21.0
pandas>=1.3.0
plotly>=5.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

---

## Team

**Ngao Labs Cohort II - Capstone Project**

| Name | Role |
|------|------|
| Dave Karanja | Lead Developer & ML Engineer |
| Jedidiah | Data Scientist |
| Pauline | UI/UX Developer |

**Duration:** March 2026 – April 2026

---

## License

- **Code:** MIT License
- **Dataset:** CC0 1.0 Universal (Public Domain)

---

## Acknowledgments

- [Five Crop Diseases Dataset](https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset) by Shubham Sharma
- Ngao Labs for capstone program support
- Google Colab for GPU resources
```
