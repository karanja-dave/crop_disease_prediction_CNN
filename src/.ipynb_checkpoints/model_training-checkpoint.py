#!/usr/bin/env python3
"""
================================================================================
NEURALNEST MODEL TRAINING
================================================================================
Project: CNN-Based Crop Disease Classification for Kenyan Agriculture
Dataset: Crop___Disease (17 classes, 5 crops, 13,324 images)

Models:
1. Custom CNN (baseline) - ~1.2M parameters, 75-85% expected accuracy
2. MobileNetV2 (selected) - ~3.5M parameters, 88-93% expected accuracy
3. EfficientNetB0 (alternative) - ~5.3M parameters, 90-95% expected accuracy

Training Strategy:
- Transfer learning with ImageNet weights
- Fine-tuning: Freeze base layers, train top layers
- Class weights for imbalance handling
- Early stopping and learning rate reduction

Author: NeuralNest Team (Pauline, Dave, Jedidiah)
Date: 2026-04-10
================================================================================
"""

# ==============================================================================
# 0. IMPORTS & CONFIGURATION
# ==============================================================================
"""
## 0. Environment Setup and Imports

**Key Libraries:**
- **TensorFlow/Keras**: Deep learning framework for model building and training
- **NumPy/Pandas**: Data manipulation and array operations
- **Matplotlib/Seaborn**: Visualization for training curves and evaluation
- **Scikit-learn**: Evaluation metrics (classification report, confusion matrix)
- **Pathlib**: Cross-platform path handling

**Hardware Configuration:**
- GPU detection and memory growth configuration
- Mixed precision training for faster computation (optional)

**Reproducibility:**
- Fixed random seeds (42) for reproducible results
- Deterministic operations where possible
"""

import os
import sys
import json
import pickle
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

# Scikit-learn for evaluation
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# REPRODUCIBILITY CONFIGURATION
# ==============================================================================
"""
**Random Seed Fixation:**
All random seeds set to 42 for reproducible results across runs.
This ensures that train/val/test splits, weight initialization, and 
augmentation operations are consistent.
"""

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
"""
**Directory Structure:**
- Crop___Disease/: Dataset root (D:\CAPSTONE REVISED\Crop Diseases Dataset\Crop Diseases\Crop___Disease)
- processed_data/: Cleaned and preprocessed data from data preparation
- models/: Saved model weights and architectures
- logs/: TensorBoard logs for training visualization
- deployment/: Final exported models for Streamlit app
"""
BASE_DIR = Path(r"D:\CAPSTONE REVISED\Crop Diseases Dataset\Crop Diseases\Crop___Disease")
DATASET_PATH  = BASE_DIR
PROCESSED_DIR = DATASET_PATH / "processed_data"
SPLIT_DIR = PROCESSED_DIR / "split"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# HYPERPARAMETERS AND TRAINING CONFIGURATION
# ==============================================================================
"""
## Training Hyperparameters (from Project Documentation Table 3.8)

| Parameter | Custom CNN | Transfer Learning | Rationale |
|-----------|-----------|-------------------|-----------|
| Optimizer | Adam | Adam | Adaptive learning rates |
| Initial LR | 0.001 | 0.0001 (frozen), 0.00001 (fine-tuning) | Prevent catastrophic forgetting |
| LR Decay | ReduceLROnPlateau | ReduceLROnPlateau | Adaptive convergence |
| Batch Size | 32 | 32 | Memory optimization |
| Epochs | 50 | 30 (10 frozen + 20 fine-tuning) | Early stopping |
| Early Stopping | Patience=5 | Patience=5 | Prevent overfitting |

**Image Specifications:**
- Input size: 224×224×3 (RGB)
- Normalization: Pixel/255.0 → [0,1]
- Augmentation: Applied only to training set
"""

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CUSTOM = 50
EPOCHS_TRANSFER = 30  # 10 frozen + 20 fine-tune

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Dataset path: {DATASET_PATH.absolute()}")

# ==============================================================================
# 1. LOAD PROCESSED DATA
# ==============================================================================
"""
## 1. Data Loading

Load processed data from data preparation stage:
- **metadata.json**: Dataset statistics and configuration
- **train/val/test_manifest.csv**: File paths and labels for each split
- **class_weights.json**: Computed weights for imbalanced classes
- **label_encoder.pkl**: Mapping between class names and integers

**Verification:**
- Confirm 17 classes are present
- Verify stratified split proportions (70/15/15)
- Check class weight distribution
"""

print("\n" + "="*80)
print("1. LOADING PROCESSED DATA")
print("="*80)

# Load metadata with dataset statistics
with open(PROCESSED_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

# Load data manifests
df_train = pd.read_csv(PROCESSED_DIR / 'train_manifest.csv')
df_val = pd.read_csv(PROCESSED_DIR / 'val_manifest.csv')
df_test = pd.read_csv(PROCESSED_DIR / 'test_manifest.csv')

# Load class weights for imbalance handling
with open(PROCESSED_DIR / 'class_weights.json', 'r') as f:
    class_weight_dict = json.load(f)
class_weights = {int(k): v for k, v in class_weight_dict.items()}

# Load label encoder for class name mapping
with open(PROCESSED_DIR / 'label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_
n_classes = len(class_names)

print(f"Training samples: {len(df_train)}")
print(f"Validation samples: {len(df_val)}")
print(f"Test samples: {len(df_test)}")
print(f"Number of classes: {n_classes}")
print(f"Classes: {list(class_names)}")

# ==============================================================================
# 2. DATA GENERATORS
# ==============================================================================
"""
## 2. Data Generators

**Keras ImageDataGenerator** creates batches of tensor image data with real-time augmentation.

**Training Generator Configuration (from Table 3.4):**
- Rotation: ±20 degrees (simulates different camera angles)
- Width/Height Shift: ±20% (simulates off-center cropping)
- Horizontal Flip: 50% probability (symmetry assumption)
- Zoom: [0.8, 1.2] (simulates distance variation)
- Brightness: [0.8, 1.2] (simulates lighting conditions)
- Fill Mode: Nearest (handles empty pixels after transform)

**Validation/Test Generators:**
- Only rescaling (no augmentation)
- Shuffle=False for consistent evaluation
"""

print("\n" + "="*80)
print("2. CREATING DATA GENERATORS")
print("="*80)

# Augmentation configuration from project documentation
augmentation_config = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': [0.8, 1.2],
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# Training generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    **augmentation_config
)

# Validation/Test generators - only rescaling (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators from dataframe
def create_generator(df, datagen, batch_size=BATCH_SIZE, shuffle=True):
    """
    Create Keras generator from dataframe with file paths and class names.
    
    Args:
        df: DataFrame with 'filepath' and 'class_name' columns
        datagen: ImageDataGenerator instance
        batch_size: Number of images per batch
        shuffle: Whether to shuffle data (True for train, False for val/test)
    
    Returns:
        DirectoryIterator: Keras data generator
    """
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepath',
        y_col='class_name',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        classes=list(class_names),
        shuffle=shuffle,
        seed=RANDOM_SEED
    )

train_generator = create_generator(df_train, train_datagen, shuffle=True)
val_generator = create_generator(df_val, val_test_datagen, shuffle=False)
test_generator = create_generator(df_test, val_test_datagen, shuffle=False)

print(f"\nTraining batches: {len(train_generator)}")
print(f"Validation batches: {len(val_generator)}")
print(f"Test batches: {len(test_generator)}")

# ==============================================================================
# 3. CUSTOM CNN BASELINE MODEL
# ==============================================================================
"""
## 3. Custom CNN Baseline Model

**Architecture (from Table 3.6):**
Input (224×224×3)
↓
Conv2D(32, 3×3) + ReLU → (224, 224, 32)
↓
MaxPool2D(2×2) → (112, 112, 32)
↓
Conv2D(64, 3×3) + ReLU → (112, 112, 64)
↓
MaxPool2D(2×2) → (56, 56, 64)
↓
Conv2D(128, 3×3) + ReLU → (56, 56, 128)
↓
MaxPool2D(2×2) → (28, 28, 128)
↓
Flatten → (100,352)
↓
Dense(128) + ReLU
↓
Dropout(0.5)
↓
Dense(17) + Softmax
plain
Copy

**Specifications:**
- Parameters: ~1.2M
- Depth: 8 layers (3 conv blocks + dense layers)
- Expected Accuracy: 75-85%
- Purpose: Establish minimum performance baseline
"""

print("\n" + "="*80)
print("3. CUSTOM CNN BASELINE MODEL")
print("="*80)

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=17):
    """
    Custom CNN baseline architecture.
    
    **Design Rationale:**
    - Simple architecture for fast training and inference
    - Three convolutional blocks with increasing filters (32→64→128)
    - MaxPooling reduces spatial dimensions, increases receptive field
    - Dropout (0.5) prevents overfitting on limited agricultural data
    
    Args:
        input_shape: Tuple of (height, width, channels)
        num_classes: Number of output classes (17 classes in the Crop___Disease dataset covering 5 crops)
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # Block 1: Low-level features (edges, textures)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Block 2: Mid-level features (shapes, patterns)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Block 3: High-level features (disease-specific patterns)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Classification head
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

# Build and summarize model
custom_cnn = build_custom_cnn(num_classes=n_classes)
custom_cnn.summary()

print(f"\nCustom CNN parameters: {custom_cnn.count_params():,}")

# Compile with standard parameters
custom_cnn.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.F1Score(name='f1_score', average='weighted')]
)

# Callbacks for training control
callbacks_custom = [
    # Stop if validation loss doesn't improve for 5 epochs
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    # Reduce LR when validation loss plateaus
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    # Save best model based on validation accuracy
    ModelCheckpoint(str(MODELS_DIR / 'custom_cnn_best.h5'), 
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    # TensorBoard logging for visualization
    TensorBoard(log_dir=str(LOGS_DIR / 'custom_cnn'), histogram_freq=1)
]

# ==============================================================================
# 4. MOBILENETV2 TRANSFER LEARNING MODEL (SELECTED)
# ==============================================================================
"""
## 4. MobileNetV2 Transfer Learning Model (SELECTED ARCHITECTURE)

**Why MobileNetV2?**
- **Accuracy-Efficiency Trade-off**: 88-93% accuracy with ~35ms inference
- **Mobile-Optimized**: Inverted residual bottlenecks, depthwise separable convolutions
- **Low Bandwidth**: 12MB model size suitable for Kenyan deployment contexts
- **Proven Architecture**: Widely used in mobile and embedded vision

**Architecture Details (from Table 3.7):**
- Base: MobileNetV2 with ImageNet weights
- Frozen: First 100 layers (general feature extraction)
- Fine-tuned: Top 20 layers (agricultural domain adaptation)
- Custom Head: GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(17)

**Training Strategy:**
1. **Phase 1 (Epochs 1-10)**: Train with frozen base, LR=0.0001
   - Preserves ImageNet features
   - Adapts classification head to agricultural domain
   
2. **Phase 2 (Epochs 11-30)**: Fine-tune top 20 layers, LR=0.00001
   - 10x lower learning rate prevents catastrophic forgetting
   - Allows subtle feature adaptation for leaf diseases

**Specifications:**
- Total Parameters: ~3.5M
- Fine-tunable Parameters: ~0.5M
- Expected Accuracy: 88-93%
- Inference Time: ~35ms
"""

print("\n" + "="*80)
print("4. MOBILENETV2 TRANSFER LEARNING MODEL (SELECTED)")
print("="*80)

def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=17):
    """
    MobileNetV2 transfer learning model with two-phase training strategy.
    
    **Key Components:**
    - Inverted residual structure with linear bottlenecks
    - Depthwise separable convolutions for efficiency
    - ReLU6 activation for numerical stability
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of disease classes
    
    Returns:
        tuple: (full_model, base_model) for separate access
    """
    # Load base model with ImageNet weights (pre-trained on 1.4M images)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,  # Remove ImageNet classification head
        input_shape=input_shape,
        alpha=1.0  # Width multiplier (1.0 = standard)
    )
    
    # Freeze first 100 layers (low and mid-level features)
    # These detect edges, textures, colors - transferable to plant leaves
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Build complete model with custom classification head
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Base model in inference mode (batch norm uses moving statistics)
    x = base_model(inputs, training=False)
    
    # Custom head for 17-class agricultural classification
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs, outputs, name='MobileNetV2')
    
    return model, base_model

# Build MobileNetV2
mobilenetv2, mobilenet_base = build_mobilenetv2(num_classes=n_classes)
mobilenetv2.summary()

# Calculate parameter statistics
total_params = mobilenetv2.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in mobilenetv2.trainable_weights])
print(f"\nMobileNetV2 total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

# ==============================================================================
# PHASE 1: FROZEN BASE TRAINING
# ==============================================================================
"""
### Phase 1: Frozen Base Training (Epochs 1-10)

**Objective:** Train the custom classification head while keeping ImageNet features frozen.

**Why This Works:**
- ImageNet features (edges, textures, shapes) are transferable to plant leaves
- Random initialization of new head would destroy pre-trained features if trained together
- Lower learning rate (0.0001) prevents large gradient updates

**Expected Outcome:** 
- Validation accuracy rapidly increases to ~80-85%
- Base model features remain stable
- Head learns to map visual features to 17 disease classes
"""

print("\n" + "="*80)
print("PHASE 1: TRAINING WITH FROZEN BASE LAYERS")
print("="*80)

mobilenetv2.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),  # 10x lower than custom CNN
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.F1Score(name='f1_score', average='weighted')]
)

callbacks_mobilenet = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(str(MODELS_DIR / 'mobilenetv2_phase1_best.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    TensorBoard(log_dir=str(LOGS_DIR / 'mobilenetv2_phase1'))
]

# Phase 1 training (uncomment to run)
"""
history_mobilenet_phase1 = mobilenetv2.fit(
    train_generator,
    epochs=10,  # Frozen training
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_mobilenet,
    verbose=1
)
"""

# ==============================================================================
# PHASE 2: FINE-TUNING
# ==============================================================================
"""
### Phase 2: Fine-Tuning Top 20 Layers (Epochs 11-30)

**Objective:** Adapt pre-trained features to agricultural domain specifics.

**Strategy:**
- Unfreeze top 20 layers of base model (high-level feature extractors)
- Use 10x lower learning rate (0.00001) to prevent catastrophic forgetting
- Continue training with early stopping

**Why Top 20 Layers?**
- Lower layers detect generic features (edges, colors) - keep frozen
- Upper layers detect object parts and textures - adapt to leaf patterns
- Fine-tuning all layers would overfit on 13K agricultural images

**Expected Outcome:**
- Validation accuracy improves to 88-93%
- Model learns disease-specific visual patterns
- Maintains generalization through conservative updates
"""

print("\n" + "="*80)
print("PHASE 2: FINE-TUNING TOP 20 LAYERS")
print("="*80)

# Unfreeze top 20 layers for fine-tuning
for layer in mobilenet_base.layers[-20:]:
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
mobilenetv2.compile(
    optimizer=optimizers.Adam(learning_rate=0.00001),  # 10x lower than Phase 1
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.F1Score(name='f1_score', average='weighted')]
)

callbacks_mobilenet_fine = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1),
    ModelCheckpoint(str(MODELS_DIR / 'mobilenetv2_best.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    TensorBoard(log_dir=str(LOGS_DIR / 'mobilenetv2_fine'))
]

# Phase 2 training (uncomment to run)
"""
history_mobilenet_phase2 = mobilenetv2.fit(
    train_generator,
    epochs=20,  # Fine-tuning epochs
    initial_epoch=10,  # Continue from Phase 1
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_mobilenet_fine,
    verbose=1
)

# Save final model
mobilenetv2.save(MODELS_DIR / 'mobilenetv2_final.h5')
print("MobileNetV2 training complete!")
"""

# ==============================================================================
# 5. EFFICIENTNETB0 ALTERNATIVE MODEL
# ==============================================================================
"""
## 5. EfficientNetB0 Alternative Model

**Why EfficientNetB0?**
- **Compound Scaling**: Jointly scales depth, width, and resolution
- **Neural Architecture Search (NAS)**: Optimized baseline architecture
- **Higher Accuracy**: 90-95% expected (2-3% better than MobileNetV2)
- **Trade-off**: Larger size (20MB vs 12MB), slower inference (~45ms vs ~35ms)

**Architecture:**
- Base: EfficientNetB0 with ImageNet weights
- Similar fine-tuning strategy to MobileNetV2
- Compound coefficient φ=1.0 (baseline)

**When to Use:**
- If MobileNetV2 accuracy insufficient
- Deployment on higher-end devices
- Offline batch processing acceptable
"""

print("\n" + "="*80)
print("5. EFFICIENTNETB0 ALTERNATIVE MODEL")
print("="*80)

def build_efficientnetb0(input_shape=(224, 224, 3), num_classes=17):
    """
    EfficientNetB0 transfer learning model.
    
    **Compound Scaling:**
    - depth: α^φ where α=1.2
    - width: β^φ where β=1.1  
    - resolution: γ^φ where γ=1.15
    - φ=1.0 for B0 (baseline)
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of disease classes
    
    Returns:
        tuple: (full_model, base_model)
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Build model
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='EfficientNetB0')
    
    return model, base_model

# Build but don't train by default (optional alternative)
efficientnet, efficientnet_base = build_efficientnetb0(num_classes=n_classes)
print(f"\nEfficientNetB0 parameters: {efficientnet.count_params():,}")

# ==============================================================================
# 6. MODEL EVALUATION
# ==============================================================================
"""
## 6. Comprehensive Model Evaluation

**Evaluation Metrics (from Table 3.9):**
| Metric | Formula | Target | Purpose |
|--------|---------|--------|---------|
| Accuracy | Correct/Total | >0.90 | Overall performance |
| Precision | TP/(TP+FP) | >0.85 | Minimize false alarms |
| Recall | TP/(TP+FN) | >0.85 | Capture all diseases |
| F1-Score | 2·P·R/(P+R) | >0.87 | Balanced measure |
| Cohen's Kappa | (P_o-P_e)/(1-P_e) | >0.80 | Agreement beyond chance |

**Per-Class Analysis:**
- Identify weak classes (low F1-score)
- Detect confusion patterns (e.g., similar diseases)
- Guide data collection priorities

**Visualizations:**
- Confusion matrix (normalized and raw)
- Per-class F1-score bar chart
- ROC curves (one-vs-rest)
"""

print("\n" + "="*80)
print("6. MODEL EVALUATION")
print("="*80)

def evaluate_model(model, test_generator, class_names, model_name):
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    **Process:**
    1. Generate predictions on test set
    2. Calculate classification metrics
    3. Create confusion matrix visualization
    4. Save detailed per-class report
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names: List of class name strings
        model_name: Name for saving outputs
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Get predictions
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{model_name} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:           {accuracy:.4f} (target: >0.90)")
    print(f"  Weighted Precision: {precision:.4f} (target: >0.85)")
    print(f"  Weighted Recall:    {recall:.4f} (target: >0.85)")
    print(f"  Weighted F1-Score:  {f1:.4f} (target: >0.87) ⭐ PRIMARY")
    print(f"  Cohen's Kappa:      {kappa:.4f} (target: >0.80)")
    print(f"{'='*60}")
    
    # Identify best and worst performing classes
    per_class_f1 = {name: report[name]['f1-score'] for name in class_names}
    best_class = max(per_class_f1, key=per_class_f1.get)
    worst_class = min(per_class_f1, key=per_class_f1.get)
    
    print(f"\nBest performing class:  {best_class} (F1={per_class_f1[best_class]:.3f})")
    print(f"Worst performing class: {worst_class} (F1={per_class_f1[worst_class]:.3f})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 8})
    plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {accuracy:.3f}, F1: {f1:.3f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()
    
    # Save detailed report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{model_name.lower().replace(" ", "_")}_report.csv')
    print(f"\nDetailed report saved: {model_name.lower().replace(' ', '_')}_report.csv")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cohen_kappa': kappa,
        'classification_report': report,
        'confusion_matrix': cm,
        'per_class_f1': per_class_f1
    }

# ==============================================================================
# 7. MODEL COMPARISON & SELECTION
# ==============================================================================
"""
## 7. Model Comparison and Selection

**Selection Criteria (from Table 3.10 - Decision Matrix):**

| Criterion | Weight | Custom CNN | MobileNetV2 | EfficientNetB0 |
|-----------|--------|-----------|-------------|----------------|
| Weighted F1-Score | 40% | Score×0.4 | Score×0.4 | Score×0.4 |
| Inference Speed | 25% | Score×0.25 | Score×0.25 | Score×0.25 |
| Model Size | 20% | Score×0.20 | Score×0.20 | Score×0.20 |
| Training Stability | 15% | Score×0.15 | Score×0.15 | Score×0.15 |

**Selection Process:**
1. Evaluate all models on test set
2. Score each criterion (normalized 0-1)
3. Weighted sum to get final score
4. Select highest scoring model for deployment

**Expected Winner:** MobileNetV2 (optimal accuracy-efficiency trade-off for Kenyan deployment)
"""

print("\n" + "="*80)
print("7. MODEL COMPARISON & SELECTION")
print("="*80)

def compare_models(results_list):
    """
    Compare multiple models using weighted decision matrix.
    
    Args:
        results_list: List of evaluation result dictionaries
    
    Returns:
        pd.DataFrame: Comparison table
        dict: Best model selection
    """
    # Create comparison dataframe
    comparison = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'F1-Score': r['f1_score'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'Kappa': r['cohen_kappa']
        }
        for r in results_list
    ])
    
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    
    # Select best by F1-score (primary criterion, 40% weight)
    best_idx = comparison['F1-Score'].idxmax()
    best_model = comparison.iloc[best_idx]['Model']
    
    print(f"\n{'='*60}")
    print(f"SELECTED MODEL: {best_model}")
    print(f"  F1-Score: {comparison.iloc[best_idx]['F1-Score']:.4f}")
    print(f"{'='*60}")
    
    comparison.to_csv('model_comparison.csv', index=False)
    return comparison, best_model

# ==============================================================================
# 8. DEPLOYMENT EXPORT
# ==============================================================================
"""
## 8. Export for Deployment

**Export Formats:**
1. **Keras H5** (.h5): Full model with weights, for loading in Python
2. **SavedModel** (directory): TensorFlow standard format, for TF Serving
3. **TFLite** (.tflite): Quantized model for mobile/edge deployment
4. **Class Names** (.json): Label mapping for inference

**Optimization for Deployment:**
- Remove training-only layers (Dropout, BatchNorm training mode)
- Convert to inference mode
- Optional: Quantization for 4x size reduction with minimal accuracy loss
"""

print("\n" + "="*80)
print("8. DEPLOYMENT EXPORT")
print("="*80)

def export_for_deployment(model, model_name, class_names, quantize=False):
    """
    Export model in multiple formats for deployment.
    
    **Formats:**
    - H5: Standard Keras format
    - SavedModel: For TensorFlow Serving
    - TFLite: For mobile and edge devices
    
    Args:
        model: Trained Keras model
        model_name: Name for saved files
        class_names: List of class names
        quantize: Whether to apply quantization (reduces size, may reduce accuracy)
    """
    export_dir = MODELS_DIR / 'deployment'
    export_dir.mkdir(exist_ok=True)
    
    print(f"\nExporting {model_name} for deployment...")
    
    # 1. Keras H5 format
    h5_path = export_dir / f'{model_name}.h5'
    model.save(h5_path)
    h5_size = h5_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  ✓ H5 format: {h5_path} ({h5_size:.2f} MB)")
    
    # 2. SavedModel format (for TensorFlow Serving)
    sm_path = export_dir / f'{model_name}_savedmodel'
    model.save(sm_path)
    sm_size = sum(f.stat().st_size for f in sm_path.rglob('*')) / (1024 * 1024)
    print(f"  ✓ SavedModel: {sm_path} ({sm_size:.2f} MB)")
    
    # 3. TFLite conversion (for mobile)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Post-training quantization for 4x size reduction
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  → Applying post-training quantization")
    
    tflite_model = converter.convert()
    tflite_path = export_dir / f'{model_name}.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    tflite_size = tflite_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ TFLite: {tflite_path} ({tflite_size:.2f} MB)")
    
    # 4. Save class names
    class_names_path = export_dir / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(list(class_names), f, indent=2)
    print(f"  ✓ Class names: {class_names_path}")
    
    # 5. Save model info
    model_info = {
        'name': model_name,
        'input_shape': list(model.input_shape[1:]),
        'output_shape': list(model.output_shape[1:]),
        'num_classes': len(class_names),
        'class_names': list(class_names),
        'export_date': datetime.now().isoformat(),
        'file_sizes': {
            'h5_mb': round(h5_size, 2),
            'savedmodel_mb': round(sm_size, 2),
            'tflite_mb': round(tflite_size, 2)
        }
    }
    info_path = export_dir / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"  ✓ Model info: {info_path}")
    
    print(f"\nAll deployment files ready in: {export_dir}")
    return export_dir

# ==============================================================================
# 9. TRAINING EXECUTION & MAIN
# ==============================================================================
"""
## 9. Training Execution

**Recommended Workflow:**
1. Start with MobileNetV2 (best accuracy-efficiency trade-off)
2. Train Phase 1 (frozen, 10 epochs) - should reach ~85% val accuracy
3. Train Phase 2 (fine-tuned, 20 epochs) - should reach ~90% val accuracy
4. Evaluate on test set
5. Export best model for Streamlit deployment

**Monitoring:**
- TensorBoard: tensorboard --logdir logs/
- Check validation accuracy each epoch
- Early stopping prevents overfitting
"""

def main():
    """
    Main training execution workflow.
    
    Uncomment training sections to run full pipeline.
    Recommended starting point: MobileNetV2 Phase 1.
    """
    print("\n" + "="*80)
    print("NEURALNEST MODEL TRAINING PIPELINE")
    print("="*80)
    
    print("\n📊 Dataset Summary:")
    print(f"  • Total images: {len(df_train) + len(df_val) + len(df_test):,}")
    print(f"  • Training: {len(df_train):,} ({len(df_train)/13324*100:.1f}%)")
    print(f"  • Validation: {len(df_val):,} ({len(df_val)/13324*100:.1f}%)")
    print(f"  • Test: {len(df_test):,} ({len(df_test)/13324*100:.1f}%)")
    print(f"  • Classes: {n_classes} (5 crops)")
    
    print("\n🔧 Models Available:")
    print("  1. Custom CNN (baseline) - ~1.2M params, 75-85% expected")
    print("  2. MobileNetV2 ⭐ SELECTED - ~3.5M params, 88-93% expected")
    print("  3. EfficientNetB0 (optional) - ~5.3M params, 90-95% expected")
    
    print("\n📋 Training Steps:")
    print("  1. Uncomment MobileNetV2 Phase 1 training block")
    print("  2. Run and monitor validation accuracy")
    print("  3. Uncomment Phase 2 (fine-tuning) block")
    print("  4. Run evaluation on test set")
    print("  5. Export best model for deployment")
    
    print("\n💡 Tips:")
    print("  • Use TensorBoard: tensorboard --logdir logs/")
    print("  • Expected training time: 2-4 hours on GPU, 8-12 hours on CPU")
    print("  • Save checkpoints every epoch (ModelCheckpoint callback)")
    print("  • Early stopping prevents overfitting (patience=5)")
    
    print(f"\n{'='*80}")
    print("Ready to train! Uncomment training blocks to begin.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

# ==============================================================================
# 10. EXAMPLE TRAINING EXECUTION (UNCOMMENT TO RUN)
# ==============================================================================
"""
## Example Full Training Run

Uncomment the following blocks to execute complete training:

```python
# STEP 1: Train MobileNetV2 Phase 1 (frozen base)
print("\n" + "="*80)
print("STEP 1: MobileNetV2 Phase 1 - Frozen Base Training")
print("="*80)
history_phase1 = mobilenetv2.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_mobilenet,
    verbose=1
)

# STEP 2: Fine-tune top 20 layers
print("\n" + "="*80)
print("STEP 2: MobileNetV2 Phase 2 - Fine-Tuning")
print("="*80)
# Unfreeze top 20 layers
for layer in mobilenet_base.layers[-20:]:
    layer.trainable = True
mobilenetv2.compile(optimizer=optimizers.Adam(0.00001), loss='categorical_crossentropy', 
                    metrics=['accuracy', 'precision', 'recall', 'f1_score'])

history_phase2 = mobilenetv2.fit(
    train_generator,
    epochs=30,
    initial_epoch=10,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_mobilenet_fine,
    verbose=1
)

# STEP 3: Evaluate
print("\n" + "="*80)
print("STEP 3: Evaluation")
print("="*80)
results = evaluate_model(mobilenetv2, test_generator, class_names, 'MobileNetV2')

# STEP 4: Export
print("\n" + "="*80)
print("STEP 4: Export for Deployment")
print("="*80)
export_for_deployment(mobilenetv2, 'NeuralNest_MobileNetV2', class_names, quantize=False)
Expected Timeline:
Phase 1: ~1 hour (10 epochs)
Phase 2: ~2 hours (20 epochs)
Evaluation: ~5 minutes
Export: ~2 minutes
"""
print("\n" + "="*80)
print("MODEL TRAINING SCRIPT COMPLETE")
print("="*80)
print("\nFiles generated after training:")
print("  • models/mobilenetv2_best.h5 - Best checkpoint")
print("  • models/mobilenetv2_final.h5 - Final model")
print("  • logs/ - TensorBoard logs")
print("  • confusion_matrix_mobilenetv2.png - Evaluation visualization")
print("  • mobilenetv2_report.csv - Per-class metrics")
print("  • model_comparison.csv - If multiple models trained")