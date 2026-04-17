# Crop Disease Classification

Deep learning project for multi-class classification of crop diseases using leaf images. Built as a capstone project under the NgaoLabs Data Science Training Program.

Authors
- Pauline Onyango
- Jedidiah Waweru
- Dave Karanja

Deployment

Live Application:
https://cropdiseasespredictioncnn-d2sxrpgrmhfgxbcotwiczz.streamlit.app/

---

## Executive Summary

- **Problem Type:** Multi-class Image Classification (14 classes)  
- **Input Data:** Leaf images (RGB)  
- **Planned Models:** Custom CNN, Transfer Learning (MobileNetV2 / EfficientNetB0)  
- **Primary Metric:** Weighted F1-score (class imbalance sensitivity)  
- **Deployment Target:** Streamlit web application  

### Expected Results & Insights
- Transfer learning models expected to outperform custom CNN due to pre-trained feature extraction  
- Anticipated challenges with generalization due to controlled dataset conditions  
- Model interpretability via Grad-CAM to highlight disease-relevant regions  

---

## Problem Context

Crop diseases reduce agricultural productivity and are often diagnosed late due to limited access to expertise. Manual inspection is slow and inconsistent. An automated image-based classification system enables faster and more scalable diagnosis.

---

## Approach

### Data Preparation
- Image resizing to 224×224  
- Pixel normalization  
- Stratified train/validation/test split (70/15/15)  
- Data augmentation: rotation, flipping, zoom, brightness adjustment  

### Exploratory Data Analysis (Planned)
- Class distribution assessment  
- Sample visualization per class  
- Image quality and size consistency checks  
- Duplicate detection  

### Feature Engineering
- Implicit feature extraction via CNN architectures  
- No manual feature engineering required  

---

## Modeling

### Models to be Evaluated
- Custom CNN (baseline)  
- MobileNetV2 (transfer learning)  
- EfficientNetB0 (transfer learning)  

### Model Selection Rationale
- Custom CNN establishes baseline performance  
- Transfer learning leverages ImageNet features for improved generalization and efficiency  

### Training Strategy
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Epochs: 20–50 with early stopping  
- Callbacks: ReduceLROnPlateau, ModelCheckpoint  

---

## Evaluation Plan

Metrics:
- Accuracy  
- Precision (per-class, macro)  
- Recall (per-class, macro)  
- **F1-score (primary metric)**  
- Confusion Matrix  

Model selection based on:
1. Weighted F1-score  
2. Inference speed  
3. Model size  

Validation:
- Hold-out test set  
- Optional cross-validation (resource dependent)  

---

## Dataset

- **Source:** Kaggle – New Plant Diseases Dataset  
- **Size:** 13,324 images  
- **Classes:** 14 (healthy + diseased states across corn, potato, wheat)  
- **Format:** RGB images, varying original sizes  

---

## Deployment Plan

**Platform:** Streamlit  

Features:
- Image upload interface  
- Predicted class + confidence score  
- Top-3 predictions  
- Rule-based advisory messages  
- Optional Grad-CAM visualization  

---

## Project Structure
```
Crop-Disease-Classification/
│
├── data/
├── notebooks/
├── models/
├── app/
├── src/
├── README.md
└── requirements.txt
```

---

## Reproducibility (Planned)

1. Clone the repository: 
```
git clone git@github.com:karanja-dave/crop_disease_prediction_CNN.git
```

2. Navigate to project folder:
```
cd crop_disease_prediction_CNN
```

3. Create virtual enviroment
```bash
python -m venv .venv
```

4. Activate the virtual enviroment
```bash
.venv\Scripts\activate
```
5. Install dependencies:  
```bash
pip install -r requirements.txt

