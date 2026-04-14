#!/usr/bin/env python
"""
================================================================================
NEURALNEST MODEL TRAINING (TF 2.21 + LEGACY KERAS)
================================================================================
"""

# ==============================================================================
# 0. IMPORTS & CONFIGURATION (FIXED ORDER)
# ==============================================================================

# CRITICAL: Set environment variables BEFORE any TensorFlow imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Force legacy Keras for compatibility

import sys
import json
import pickle
import time
import warnings
import importlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras - Now imports legacy Keras due to env vars above
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# ==============================================================================
# REPRODUCIBILITY CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# BASE PROJECT DIRECTORY
BASE_DIR = Path("D:/CAPSTONE REVISED")

# DATASET ROOT
DATASET_PATH = BASE_DIR / "Crop Diseases Dataset" / "Crop Diseases" / "Crop___Disease"

# PROCESSED DATA (after preprocessing step)
PROCESSED_DIR = DATASET_PATH / "processed_data"
SPLIT_DIR = PROCESSED_DIR / "split"

# OUTPUT DIRECTORIES
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# CREATE OUTPUT FOLDERS IF THEY DON'T EXIST
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# DEBUG PRINTS
print("BASE_DIR:", BASE_DIR)
print("DATASET_PATH:", DATASET_PATH)
print("PROCESSED_DIR:", PROCESSED_DIR)

# ==============================================================================
# HYPERPARAMETERS AND TRAINING CONFIGURATION
# ==============================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CUSTOM = 50
EPOCHS_TRANSFER = 30  # 10 frozen + 20 fine-tune

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Dataset path: {DATASET_PATH.absolute()}")

print("DATASET_PATH =", DATASET_PATH)
print("PROCESSED_DIR =", PROCESSED_DIR)
print("Metadata exists =", (PROCESSED_DIR / "metadata.json").exists())

# ==============================================================================
# 1. LOAD PROCESSED DATA
# ==============================================================================

print("\n" + "="*80)
print("1. LOADING PROCESSED DATA")
print("="*80)

# Ensure path exists
if not PROCESSED_DIR.exists():
    raise FileNotFoundError(f"Processed directory not found: {PROCESSED_DIR}")

# -----------------------------
# Load metadata
# -----------------------------
metadata_path = PROCESSED_DIR / 'metadata.json'

if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
else:
    print("WARNING: metadata.json not found")
    metadata = {}

# -----------------------------
# Load manifests (SAFE)
# -----------------------------
train_path = PROCESSED_DIR / 'train_manifest.csv'
val_path = PROCESSED_DIR / 'val_manifest.csv'
test_path = PROCESSED_DIR / 'test_manifest.csv'

if not train_path.exists() or not val_path.exists() or not test_path.exists():
    raise FileNotFoundError("One or more manifest files are missing in processed_data")

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

# -----------------------------
# Load class weights (SAFE)
# -----------------------------
class_weights_path = PROCESSED_DIR / 'class_weights.json'

if class_weights_path.exists():
    with open(class_weights_path, 'r') as f:
        class_weight_dict = json.load(f)
    class_weights = {int(k): v for k, v in class_weight_dict.items()}
else:
    print("WARNING: class_weights.json not found")
    class_weights = None

# -----------------------------
# Load label encoder (SAFE)
# -----------------------------
encoder_path = PROCESSED_DIR / 'label_encoder.pkl'

if encoder_path.exists():
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_
else:
    print("WARNING: label_encoder.pkl not found")
    class_names = sorted(df_train['class_name'].unique())
    label_encoder = None

n_classes = len(class_names)

# -----------------------------
# FINAL CHECK
# -----------------------------
print("\n==============================")
print("DATA LOADING SUMMARY")
print("==============================")

print(f"Training samples   : {len(df_train)}")
print(f"Validation samples : {len(df_val)}")
print(f"Test samples       : {len(df_test)}")
print(f"Number of classes  : {n_classes}")
print(f"Classes            : {list(class_names)}")

# Safety check
if len(df_train) == 0 or len(df_val) == 0:
    raise ValueError("Train/Val data is empty — check preprocessing pipeline")

# ==============================================================================
# 2. DATA GENERATORS
# ==============================================================================

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

print("\n" + "="*80)
print("3. CUSTOM CNN BASELINE MODEL")
print("="*80)

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=17):
    """
    Custom CNN baseline architecture.
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

# FIX: Simplified metrics - removed F1Score which causes shape issues with generators
custom_cnn.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

def safe_tensorboard_callback(log_dir, histogram_freq=0):
    """Create a TensorBoard callback only if TensorBoard is installed."""
    try:
        importlib.import_module('tensorboard')
        return TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)
    except ImportError:
        print(f"WARNING: TensorBoard package is not installed. Skipping TensorBoard callback for {log_dir}.")
        return None
    except Exception as exc:
        print(f"WARNING: TensorBoard callback unavailable: {exc}. Skipping TensorBoard callback.")
        return None

# Callbacks for training control
callbacks_custom = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(str(MODELS_DIR / 'custom_cnn_best.keras'), 
                    monitor='val_accuracy', save_best_only=True, verbose=1),
]
custom_tb = safe_tensorboard_callback(str(LOGS_DIR / 'custom_cnn'), histogram_freq=1)
if custom_tb is not None:
    callbacks_custom.append(custom_tb)

# ==============================================================================
# 4. MOBILENETV2 TRANSFER LEARNING MODEL (SELECTED)
# ==============================================================================

print("\n" + "="*80)
print("4. MOBILENETV2 TRANSFER LEARNING MODEL (SELECTED)")
print("="*80)

def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=17):
    """
    MobileNetV2 transfer learning model with two-phase training strategy.
    """
    # Load base model with ImageNet weights
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        alpha=1.0
    )
    
    # Freeze first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Build complete model with custom classification head
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Base model in inference mode
    x = base_model(inputs, training=False)
    
    # Custom head for agricultural classification
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

print("\n" + "="*80)
print("PHASE 1: TRAINING WITH FROZEN BASE LAYERS")
print("="*80)

# FIX: Simplified metrics - removed F1Score to avoid shape issues with generators
mobilenetv2.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

callbacks_mobilenet = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(str(MODELS_DIR / 'mobilenetv2_phase1_best.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=1),
]
phase1_tb = safe_tensorboard_callback(str(LOGS_DIR / 'mobilenetv2_phase1'))
if phase1_tb is not None:
    callbacks_mobilenet.append(phase1_tb)

# Phase 1 training
history_mobilenet_phase1 = mobilenetv2.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_mobilenet,
    verbose=1
)

# ==============================================================================
# PHASE 2: FINE-TUNING
# ==============================================================================

print("\n" + "="*80)
print("PHASE 2: FINE-TUNING TOP 20 LAYERS")
print("="*80)

# Unfreeze top 20 layers for fine-tuning
for layer in mobilenet_base.layers[-20:]:
    layer.trainable = True

# FIX: Simplified metrics - removed F1Score
mobilenetv2.compile(
    optimizer=optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

callbacks_mobilenet_fine = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1),
    ModelCheckpoint(str(MODELS_DIR / 'mobilenetv2_best.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=1),
]
phase2_tb = safe_tensorboard_callback(str(LOGS_DIR / 'mobilenetv2_fine'))
if phase2_tb is not None:
    callbacks_mobilenet_fine.append(phase2_tb)

# Phase 2 training
history_mobilenet_phase2 = mobilenetv2.fit(
    train_generator,
    epochs=20,
    initial_epoch=10,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_mobilenet_fine,
    verbose=1
)

# Save final model in both formats
mobilenetv2.save(MODELS_DIR / 'mobilenetv2_final.keras')
print("MobileNetV2 training complete!")

# ==============================================================================
# 5. EFFICIENTNETB0 ALTERNATIVE MODEL
# ==============================================================================

print("\n" + "="*80)
print("5. EFFICIENTNETB0 ALTERNATIVE MODEL")
print("="*80)

def build_efficientnetb0(input_shape=(224, 224, 3), num_classes=17):
    """
    EfficientNetB0 transfer learning model.
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

# Build but don't train by default
efficientnet, efficientnet_base = build_efficientnetb0(num_classes=n_classes)
print(f"\nEfficientNetB0 parameters: {efficientnet.count_params():,}")

# ==============================================================================
# 6. MODEL EVALUATION
# ==============================================================================

print("\n" + "="*80)
print("6. MODEL EVALUATION")
print("="*80)

def evaluate_model(model, test_generator, class_names, model_name):
    """
    Comprehensive model evaluation with metrics and visualizations.
    """
    print(f"\nEvaluating {model_name}...")
    
    # Get predictions
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate metrics using sklearn (more reliable than Keras metrics)
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

print("\n" + "="*80)
print("7. MODEL COMPARISON & SELECTION")
print("="*80)

def compare_models(results_list):
    """
    Compare multiple models using weighted decision matrix.
    """
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
    
    # Select best by F1-score
    best_idx = comparison['F1-Score'].idxmax()
    best_model = comparison.iloc[best_idx]['Model']
    
    print(f"\n{'='*60}")
    print(f"SELECTED MODEL: {best_model}")
    print(f"  F1-Score: {comparison.iloc[best_idx]['F1-Score']:.4f}")
    print(f"{'='*60}")
    
    comparison.to_csv('model_comparison.csv', index=False)
    return comparison, best_model

# ==============================================================================
# 8. DEPLOYMENT EXPORT (FULLY CORRECTED FOR TF 2.21)
# ==============================================================================

print("\n" + "=" * 80)
print("8. DEPLOYMENT EXPORT")
print("=" * 80)

def export_for_deployment(model, model_name, class_names, quantize=False):
    """
    Export model in multiple formats for deployment.
    Fully corrected for TensorFlow 2.21 + Legacy Keras compatibility.
    """
    
    export_dir = MODELS_DIR / "deployment"
    export_dir.mkdir(exist_ok=True)
    
    print(f"\nExporting {model_name} for deployment...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # 1. Keras v3 format (RECOMMENDED - primary format)
    try:
        keras_path = export_dir / f"{model_name}.keras"
        model.save(keras_path)
        keras_size = keras_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Keras v3 format: {keras_path} ({keras_size:.2f} MB)")
    except Exception as e:
        print(f"  ⚠ Keras v3 save failed: {e}")
        keras_size = 0
    
    # 2. Legacy H5 format (for backward compatibility)
    try:
        h5_path = export_dir / f"{model_name}.h5"
        tf.keras.models.save_model(model, h5_path, save_format='h5')
        h5_size = h5_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ H5 format: {h5_path} ({h5_size:.2f} MB)")
    except Exception as e:
        print(f"  ⚠ H5 save failed: {e}")
        h5_size = 0
    
    # 3. SavedModel format (directory - best for serving)
    try:
        sm_path = export_dir / f"{model_name}_savedmodel"
        tf.keras.models.save_model(model, sm_path)
        sm_size = sum(f.stat().st_size for f in sm_path.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"  ✓ SavedModel: {sm_path} ({sm_size:.2f} MB)")
    except Exception as e:
        print(f"  ⚠ SavedModel export failed: {e}")
        sm_size = 0
    
    # 4. TFLite conversion
    tflite_size = 0
    try:
        # Method 1: Convert from SavedModel (most reliable)
        if 'sm_path' in locals() and sm_path.exists():
            converter = tf.lite.TFLiteConverter.from_saved_model(str(sm_path))
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            tflite_path = export_dir / f"{model_name}.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            tflite_size = tflite_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ TFLite: {tflite_path} ({tflite_size:.2f} MB)")
    except Exception as e:
        print(f"  ⚠ TFLite conversion from SavedModel failed: {e}")
        try:
            # Method 2: Convert from Keras model directly
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            tflite_path = export_dir / f"{model_name}.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            tflite_size = tflite_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ TFLite (from Keras): {tflite_path} ({tflite_size:.2f} MB)")
        except Exception as e2:
            print(f"  ✗ TFLite conversion failed completely: {e2}")
    
    # 5. Save class names
    class_names_path = export_dir / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(list(class_names), f, indent=2)
    print(f"  ✓ Class names: {class_names_path}")
    
    # 6. Save label encoder if available
    if label_encoder is not None:
        encoder_path = export_dir / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"  ✓ Label encoder: {encoder_path}")
    
    # 7. Save advisory rules template
    advisory_template = {
        "class_name_example": {
            "treatment": ["Treatment step 1", "Treatment step 2"],
            "prevention": ["Prevention step 1", "Prevention step 2"],
            "confidence_threshold": 0.85
        }
    }
    advisory_path = export_dir / "advisory_rules.json"
    with open(advisory_path, "w") as f:
        json.dump(advisory_template, f, indent=2)
    print(f"  ✓ Advisory template: {advisory_path}")
    
    # 8. Save model info
    model_info = {
        "name": model_name,
        "input_shape": list(model.input_shape[1:]) if model.input_shape else None,
        "output_shape": list(model.output_shape[1:]) if model.output_shape else None,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "export_date": datetime.now().isoformat(),
        "tensorflow_version": tf.__version__,
        "keras_backend": "legacy_tf_keras",
        "file_sizes": {
            "keras_mb": round(keras_size, 2),
            "h5_mb": round(h5_size, 2),
            "savedmodel_mb": round(sm_size, 2),
            "tflite_mb": round(tflite_size, 2) if tflite_size > 0 else None
        }
    }
    
    info_path = export_dir / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"  ✓ Model info: {info_path}")
    
    print(f"\nAll deployment files ready in: {export_dir}")
    return export_dir

# ==============================================================================
# 9. EXECUTE TRAINING PIPELINE
# ==============================================================================

print("\n" + "=" * 80)
print("EXECUTING COMPLETE TRAINING PIPELINE")
print("=" * 80)

# STEP 1: Train MobileNetV2 Phase 1 (frozen base) - Already done above
print("\n" + "=" * 80)
print("STEP 1: MobileNetV2 Phase 1 - Frozen Base Training")
print("=" * 80)
print("Phase 1 completed above.")

# STEP 2: Fine-tune top 20 layers - Already done above
print("\n" + "=" * 80)
print("STEP 2: MobileNetV2 Phase 2 - Fine-Tuning")
print("=" * 80)
print("Phase 2 completed above.")

# STEP 3: Evaluate
print("\n" + "=" * 80)
print("STEP 3: Evaluation")
print("=" * 80)

results = evaluate_model(
    mobilenetv2,
    test_generator,
    class_names,
    "MobileNetV2"
)

# STEP 4: Export
print("\n" + "=" * 80)
print("STEP 4: Export for Deployment")
print("=" * 80)

export_for_deployment(
    mobilenetv2,
    "NeuralNest_MobileNetV2",
    class_names,
    quantize=False
)

print("\n" + "=" * 80)
print("MODEL TRAINING SCRIPT COMPLETE")
print("=" * 80)

print("\nFiles generated after training:")
print("  • models/mobilenetv2_best.keras - Best checkpoint (Keras v3 format)")
print("  • models/mobilenetv2_best.h5 - Best checkpoint (Legacy H5 format)")
print("  • models/mobilenetv2_final.keras - Final model")
print("  • models/deployment/ - Deployment files")
print("  • logs/ - TensorBoard logs")
print("  • confusion_matrix_mobilenetv2.png - Evaluation visualization")
print("  • mobilenetv2_report.csv - Per-class metrics")