# Crop Disease Classification - Data Preparation Notebook
# Project: NeuralNest - AI-Powered Crop Disease Detection for Kenyan Agriculture
# Dataset: New Plant Diseases Dataset (13,324 images across 14 classes)

# ==============================================================================
# TABLE OF CONTENTS
# ==============================================================================
# 1. Environment Setup & Library Imports
# 2. Data Loading & Initial Inspection
# 3. Exploratory Data Analysis (EDA)
#    3.1 Dataset Overview & Statistics
#    3.2 Class Distribution Analysis
#    3.3 Image Quality Assessment
#    3.4 Visual Data Exploration
# 4. Data Cleaning & Quality Assurance
#    4.1 Duplicate Detection & Removal
#    4.2 Corrupted Image Handling
#    4.3 Outlier Detection
# 5. Feature Engineering
#    5.1 Image Preprocessing Pipeline
#    5.2 Data Augmentation Strategy
#    5.3 Stratified Data Splitting
#    5.4 Label Encoding & Metadata Extraction
# 6. Data Export & Pipeline Serialization
# 7. Summary & Quality Report

# ==============================================================================
# 1. ENVIRONMENT SETUP & LIBRARY IMPORTS
# ==============================================================================
"""
## 1. Environment Setup

This section establishes the computational environment with all necessary 
libraries for image processing, deep learning preprocessing, and data analysis.

**Key Libraries:**
- **Core:** os, pathlib, glob, json, pickle (file operations)
- **Image Processing:** PIL (Pillow), opencv-python (cv2)
- **Data Manipulation:** numpy, pandas
- **Visualization:** matplotlib, seaborn, plotly
- **ML Preprocessing:** scikit-learn (stratified splitting, encoding)
- **Deep Learning:** tensorflow/keras (preprocessing layers)
- **Progress Tracking:** tqdm

**Hardware Requirements:**
- Minimum 8GB RAM (16GB recommended for 13K+ images)
- GPU optional but recommended for augmentation preview
"""

import os
import sys
import json
import pickle
import shutil
import random
import warnings
from pathlib import Path
from glob import glob
from collections import Counter, defaultdict
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Data manipulation
import numpy as np
import pandas as pd

# Image processing
from PIL import Image, ImageStat, ImageEnhance
import cv2

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
%matplotlib inline

# Machine Learning preprocessing
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

# Deep Learning (TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    print(f"TensorFlow version: {tf.__version__}")
    GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"GPU Available: {GPU_AVAILABLE}")
except ImportError:
    print("TensorFlow not installed. Using PIL/OpenCV for preprocessing.")
    GPU_AVAILABLE = False

# Progress tracking
from tqdm.notebook import tqdm

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if 'tf' in globals():
    tf.random.set_seed(RANDOM_SEED)

print("\n" + "="*80)
print("ENVIRONMENT SETUP COMPLETE")
print("="*80)
print(f"Working Directory: {os.getcwd()}")
print(f"Python Version: {sys.version}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*80)

# ==============================================================================
# 2. DATA LOADING & INITIAL INSPECTION
# ==============================================================================
"""
## 2. Data Loading & Initial Inspection

This section handles the ingestion of the New Plant Diseases Dataset.
According to the project documentation:
- **Source:** Kaggle (Ahmed, 2019)
- **Total Images:** 13,324
- **Format:** RGB JPEG
- **Classes:** 14 categories (Corn, Potato, Wheat - healthy and diseased)
- **License:** CC0 Public Domain

**Expected Directory Structure:**
```
new-plant-diseases-dataset/
├── train/
│   ├── Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot/
│   ├── Corn_(maize)_Common_rust_/
│   ├── ... (14 class folders)
└── valid/
    └── (same structure)
```

**Note:** The dataset comes pre-split, but we'll re-split using stratified 
sampling to ensure proper train/validation/test distribution.
"""

# Configuration
DATASET_PATH = "new-plant-diseases-dataset"  # Adjust path as needed
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")

# Output directories for processed data
PROCESSED_DIR = "processed_data"
CLEANED_DIR = os.path.join(PROCESSED_DIR, "cleaned")
AUGMENTED_DIR = os.path.join(PROCESSED_DIR, "augmented")
SPLIT_DIR = os.path.join(PROCESSED_DIR, "split")

# Create output directories
for dir_path in [PROCESSED_DIR, CLEANED_DIR, AUGMENTED_DIR, SPLIT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dir_path, subset), exist_ok=True)

print("\n" + "="*80)
print("DATA LOADING CONFIGURATION")
print("="*80)
print(f"Dataset Path: {DATASET_PATH}")
print(f"Train Directory: {TRAIN_DIR}")
print(f"Validation Directory: {VALID_DIR}")
print(f"Processed Output: {PROCESSED_DIR}")
print("="*80)

def load_dataset_structure(train_dir, valid_dir):
    """
    Load and catalog all images from the dataset directory structure.

    Returns:
        DataFrame with columns: [filepath, filename, class_name, crop_type, 
                                health_status, original_split, image_id]
    """
    data_records = []

    # Process both train and valid directories
    for split_name, split_dir in [('train', train_dir), ('valid', valid_dir)]:
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist")
            continue

        # Iterate through class folders
        for class_name in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_name)

            # Skip non-directory items
            if not os.path.isdir(class_path):
                continue

            # Parse class information
            # Format examples: "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot"
            #                  "Potato_healthy", "Wheat_septoria"

            parts = class_name.replace('_(maize)_', '_').replace('__', '_').split('_')

            # Determine crop type
            if 'Corn' in class_name or 'maize' in class_name.lower():
                crop_type = 'Corn'
            elif 'Potato' in class_name:
                crop_type = 'Potato'
            elif 'Wheat' in class_name:
                crop_type = 'Wheat'
            else:
                crop_type = 'Unknown'

            # Determine health status
            if 'healthy' in class_name.lower():
                health_status = 'Healthy'
                disease_name = 'Healthy'
            else:
                health_status = 'Diseased'
                # Extract disease name (everything after crop name)
                disease_parts = class_name.replace(crop_type, '').replace('_(maize)', '').strip('_').split('_')
                disease_name = ' '.join([p for p in disease_parts if p and p.lower() != 'healthy']).strip()

            # Get all images in this class
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            images = []
            for ext in image_extensions:
                images.extend(glob(os.path.join(class_path, ext)))

            for img_path in images:
                data_records.append({
                    'filepath': img_path,
                    'filename': os.path.basename(img_path),
                    'class_name': class_name,
                    'crop_type': crop_type,
                    'health_status': health_status,
                    'disease_name': disease_name,
                    'original_split': split_name,
                    'image_id': f"{split_name}_{class_name}_{os.path.basename(img_path)}"
                })

    return pd.DataFrame(data_records)

# Execute data loading
print("\nLoading dataset structure...")
df_raw = load_dataset_structure(TRAIN_DIR, VALID_DIR)

print("\n" + "="*80)
print("INITIAL DATASET INSPECTION")
print("="*80)
print(f"Total Records: {len(df_raw):,}")
print(f"Unique Classes: {df_raw['class_name'].nunique()}")
print(f"Unique Crops: {df_raw['crop_type'].nunique()} ({', '.join(df_raw['crop_type'].unique())})")
print(f"Original Splits: {df_raw['original_split'].value_counts().to_dict()}")
print("\nFirst 5 records:")
print(df_raw.head())
print("\nDataset Info:")
print(df_raw.info())
print("="*80)

# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
"""
## 3. Exploratory Data Analysis (EDA)

EDA is critical for understanding the dataset characteristics before modeling.
According to the project documentation (Section 3.4), we need to analyze:

1. **Class Distribution:** Check for imbalance (Table 3.2 in documentation)
2. **Image Properties:** Dimensions, color channels, file sizes
3. **Data Quality:** Duplicates, corrupted files, outliers
4. **Visual Patterns:** Sample images per class

**Key Metrics from Documentation:**
- Target: 13,324 total images
- 14 balanced classes
- 70/15/15 train/val/test split (stratified)
- Image dimensions: Variable (400×400 to 800×800), standardized to 224×224
"""

# ------------------------------------------------------------------------------
# 3.1 Dataset Overview & Statistics
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("3.1 DATASET OVERVIEW & STATISTICS")
print("="*80)

# Basic statistics
print("\n--- Class Distribution Overview ---")
class_stats = df_raw.groupby('class_name').agg({
    'filepath': 'count',
    'crop_type': 'first',
    'health_status': 'first'
}).rename(columns={'filepath': 'count'}).sort_values('count', ascending=False)

print(class_stats)

# Crop type distribution
print("\n--- Crop Type Distribution ---")
crop_dist = df_raw['crop_type'].value_counts()
print(crop_dist)

# Health status distribution
print("\n--- Health Status Distribution ---")
health_dist = df_raw['health_status'].value_counts()
print(health_dist)

# Cross-tabulation
print("\n--- Crop vs Health Status Cross-tab ---")
crosstab = pd.crosstab(df_raw['crop_type'], df_raw['health_status'])
print(crosstab)

# ------------------------------------------------------------------------------
# 3.2 Class Distribution Analysis & Visualization
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("3.2 CLASS DISTRIBUTION ANALYSIS")
print("="*80)

# Calculate imbalance metrics
class_counts = df_raw['class_name'].value_counts()
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = max_count / min_count

print(f"\nImbalance Metrics:")
print(f"  Maximum class size: {max_count}")
print(f"  Minimum class size: {min_count}")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
print(f"  Balanced dataset: {'Yes' if imbalance_ratio < 1.5 else 'Moderate' if imbalance_ratio < 3 else 'High imbalance'}")

# Visualization: Class Distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Overall class distribution (horizontal bar)
ax1 = axes[0, 0]
class_counts_sorted = class_counts.sort_values()
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(class_counts_sorted)))
bars = ax1.barh(range(len(class_counts_sorted)), class_counts_sorted.values, color=colors)
ax1.set_yticks(range(len(class_counts_sorted)))
ax1.set_yticklabels([name.replace('_', ' ')[:30] for name in class_counts_sorted.index], fontsize=9)
ax1.set_xlabel('Number of Images')
ax1.set_title('Distribution of Images Across Classes', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, val) in enumerate(class_counts_sorted.items()):
    ax1.text(val + 10, i, str(val), va='center', fontsize=8)

# Plot 2: Crop type distribution (pie chart)
ax2 = axes[0, 1]
crop_counts = df_raw['crop_type'].value_counts()
colors_pie = ['#FF9999', '#66B2FF', '#99FF99']
wedges, texts, autotexts = ax2.pie(crop_counts, labels=crop_counts.index, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90)
ax2.set_title('Distribution by Crop Type', fontsize=12, fontweight='bold')

# Plot 3: Health status by crop (stacked bar)
ax3 = axes[1, 0]
crop_health = pd.crosstab(df_raw['crop_type'], df_raw['health_status'])
crop_health.plot(kind='bar', stacked=True, ax=ax3, color=['#FF6B6B', '#4ECDC4'])
ax3.set_title('Health Status Distribution by Crop', fontsize=12, fontweight='bold')
ax3.set_xlabel('Crop Type')
ax3.set_ylabel('Number of Images')
ax3.legend(title='Health Status')
ax3.tick_params(axis='x', rotation=0)

# Plot 4: Class imbalance visualization
ax4 = axes[1, 1]
# Calculate deviation from mean
mean_count = class_counts.mean()
deviations = ((class_counts - mean_count) / mean_count * 100).sort_values()
colors_dev = ['red' if x < -20 else 'green' if x > 20 else 'gray' for x in deviations]
ax4.barh(range(len(deviations)), deviations.values, color=colors_dev, alpha=0.7)
ax4.set_yticks(range(len(deviations)))
ax4.set_yticklabels([name.replace('_', ' ')[:25] for name in deviations.index], fontsize=8)
ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax4.set_xlabel('Deviation from Mean (%)')
ax4.set_title('Class Size Deviation from Average', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved: eda_class_distribution.png")

# ------------------------------------------------------------------------------
# 3.3 Image Properties Analysis
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("3.3 IMAGE PROPERTIES ANALYSIS")
print("="*80)

def analyze_image_properties(df, sample_size=1000):
    """
    Analyze image dimensions, file sizes, and color properties.
    Uses sampling for large datasets to improve performance.
    """
    # Sample for faster processing
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=RANDOM_SEED)
        print(f"Sampling {sample_size} images for property analysis (from {len(df)} total)")
    else:
        sample_df = df

    properties = []

    print("Analyzing image properties...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        try:
            # Get file size
            file_size = os.path.getsize(row['filepath']) / 1024  # KB

            # Open and analyze image
            with Image.open(row['filepath']) as img:
                width, height = img.size
                mode = img.mode

                # Calculate aspect ratio
                aspect_ratio = width / height

                # Get color statistics (sample for large images)
                if img.size[0] * img.size[1] > 1000000:  # > 1MP
                    img_small = img.resize((224, 224))
                else:
                    img_small = img

                stat = ImageStat.Stat(img_small)

                properties.append({
                    'image_id': row['image_id'],
                    'class_name': row['class_name'],
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'file_size_kb': file_size,
                    'color_mode': mode,
                    'mean_r': stat.mean[0] if len(stat.mean) > 0 else 0,
                    'mean_g': stat.mean[1] if len(stat.mean) > 1 else 0,
                    'mean_b': stat.mean[2] if len(stat.mean) > 2 else 0,
                    'brightness': sum(stat.mean[:3]) / 3 / 255 if len(stat.mean) >= 3 else 0
                })
        except Exception as e:
            print(f"Error processing {row['filepath']}: {e}")

    return pd.DataFrame(properties)

# Analyze image properties
props_df = analyze_image_properties(df_raw)

print("\n--- Image Dimension Statistics ---")
dim_stats = props_df.groupby(['width', 'height']).size().reset_index(name='count').sort_values('count', ascending=False)
print(f"Unique dimension combinations: {len(dim_stats)}")
print("\nTop 10 most common dimensions:")
print(dim_stats.head(10))

print("\n--- File Size Statistics ---")
print(props_df['file_size_kb'].describe())

print("\n--- Aspect Ratio Statistics ---")
print(props_df['aspect_ratio'].describe())

print("\n--- Brightness Statistics ---")
print(props_df.groupby('class_name')['brightness'].mean().sort_values())

# Visualization: Image Properties
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Dimension distribution scatter
ax1 = axes[0, 0]
scatter = ax1.scatter(props_df['width'], props_df['height'], 
                     c=props_df['file_size_kb'], cmap='viridis', 
                     alpha=0.6, s=30)
ax1.set_xlabel('Width (pixels)')
ax1.set_ylabel('Height (pixels)')
ax1.set_title('Image Dimensions Distribution', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='File Size (KB)')

# Plot 2: Aspect ratio distribution
ax2 = axes[0, 1]
ax2.hist(props_df['aspect_ratio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=1.0, color='red', linestyle='--', label='Square (1:1)')
ax2.set_xlabel('Aspect Ratio (width/height)')
ax2.set_ylabel('Frequency')
ax2.set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: File size by class (box plot)
ax3 = axes[1, 0]
# Select top 8 classes for readability
top_classes = props_df['class_name'].value_counts().head(8).index
props_subset = props_df[props_df['class_name'].isin(top_classes)]
box_data = [props_subset[props_subset['class_name'] == cls]['file_size_kb'].values 
            for cls in top_classes]
bp = ax3.boxplot(box_data, labels=[cls.replace('_', '\n')[:15] for cls in top_classes])
ax3.set_ylabel('File Size (KB)')
ax3.set_title('File Size Distribution by Class (Top 8)', fontsize=12, fontweight='bold')
ax3.tick_params(axis='x', rotation=45, labelsize=8)

# Plot 4: Brightness distribution by health status
ax4 = axes[1, 1]
# Merge with health status
props_health = props_df.merge(df_raw[['image_id', 'health_status']].drop_duplicates(), 
                              on='image_id', how='left')
for status in ['Healthy', 'Diseased']:
    data = props_health[props_health['health_status'] == status]['brightness']
    ax4.hist(data, bins=30, alpha=0.5, label=status, density=True)
ax4.set_xlabel('Normalized Brightness (0-1)')
ax4.set_ylabel('Density')
ax4.set_title('Brightness Distribution by Health Status', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_image_properties.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved: eda_image_properties.png")

# ------------------------------------------------------------------------------
# 3.4 Visual Data Exploration - Sample Images
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("3.4 VISUAL DATA EXPLORATION - SAMPLE IMAGES")
print("="*80)

def plot_sample_grid(df, samples_per_class=3, figsize=(20, 24)):
    """
    Create a grid visualization showing sample images from each class.
    """
    classes = df['class_name'].unique()
    n_classes = len(classes)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_classes, samples_per_class, figure=fig, hspace=0.4, wspace=0.3)

    for i, class_name in enumerate(sorted(classes)):
        class_samples = df[df['class_name'] == class_name].sample(
            n=min(samples_per_class, len(df[df['class_name'] == class_name])),
            random_state=RANDOM_SEED
        )

        for j, (_, row) in enumerate(class_samples.iterrows()):
            ax = fig.add_subplot(gs[i, j])

            try:
                img = Image.open(row['filepath'])
                ax.imshow(img)
                ax.axis('off')

                if j == 0:
                    # Add class label on the leftmost image
                    crop = row['crop_type']
                    health = row['health_status']
                    disease = row['disease_name'] if health == 'Diseased' else 'Healthy'
                    ax.set_ylabel(f"{crop}\n{disease[:20]}", 
                                 fontsize=10, rotation=0, ha='right', va='center')

                if i == 0:
                    ax.set_title(f"Sample {j+1}", fontsize=10)

            except Exception as e:
                ax.text(0.5, 0.5, f"Error\nloading", ha='center', va='center')
                ax.axis('off')

    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('eda_sample_images.png', dpi=200, bbox_inches='tight')
    plt.show()

    return fig

print("Generating sample image grid...")
plot_sample_grid(df_raw, samples_per_class=3)
print("\nVisualization saved: eda_sample_images.png")

# ------------------------------------------------------------------------------
# 3.5 Interactive EDA with Plotly
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("3.5 INTERACTIVE VISUALIZATION")
print("="*80)

# Create interactive class distribution chart
fig = px.bar(
    x=class_counts.values,
    y=[name.replace('_', ' ') for name in class_counts.index],
    orientation='h',
    color=class_counts.values,
    color_continuous_scale='Viridis',
    title='Interactive: Class Distribution (Hover for details)',
    labels={'x': 'Number of Images', 'y': 'Class Name', 'color': 'Count'}
)
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
fig.show()

# Treemap visualization
fig2 = px.treemap(
    df_raw,
    path=['crop_type', 'health_status', 'class_name'],
    values=df_raw.groupby('class_name')['filepath'].transform('count'),
    title='Hierarchical View: Crop → Health → Class',
    color='crop_type'
)
fig2.update_layout(height=600)
fig2.show()

print("\nInteractive visualizations generated!")

# ==============================================================================
# 4. DATA CLEANING & QUALITY ASSURANCE
# ==============================================================================
"""
## 4. Data Cleaning & Quality Assurance

According to the project methodology (Section 3.3), we must ensure:
- No duplicate images (same content, different filenames)
- No corrupted or unreadable files
- Consistent image dimensions (standardized to 224×224)
- Proper color channel handling (RGB)

**Cleaning Pipeline Stages (from Table 3.3):**
1. P1: Image Loading (Format standardization)
2. P2: Dimension Verification (Min 224×224 check)
3. P3: Resizing (224×224, bicubic)
4. P4: Color Space (RGB preservation)
5. P5: Normalization (Pixel/255.0 → [0,1])
"""

# ------------------------------------------------------------------------------
# 4.1 Duplicate Detection & Removal
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("4.1 DUPLICATE DETECTION & REMOVAL")
print("="*80)

def find_duplicates(df, method='hash', sample_for_hash=5000):
    """
    Find duplicate images using perceptual hashing or file hashing.

    Methods:
    - 'hash': MD5 file hash (exact duplicates)
    - 'phash': Perceptual hash (similar images)
    """
    print(f"Detecting duplicates using {method} method...")

    if method == 'hash':
        # File hash method - exact duplicates only
        file_hashes = {}
        duplicates = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                with open(row['filepath'], 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in file_hashes:
                    duplicates.append({
                        'original': file_hashes[file_hash],
                        'duplicate': row['filepath'],
                        'hash': file_hash
                    })
                else:
                    file_hashes[file_hash] = row['filepath']
            except Exception as e:
                print(f"Error hashing {row['filepath']}: {e}")

        return duplicates

    elif method == 'phash':
        # Perceptual hash - similar images
        from PIL import Image
        import imagehash

        phashes = {}
        duplicates = []

        # Sample for speed if dataset is large
        check_df = df.sample(n=min(sample_for_hash, len(df)), random_state=RANDOM_SEED) if len(df) > sample_for_hash else df

        for idx, row in tqdm(check_df.iterrows(), total=len(check_df)):
            try:
                with Image.open(row['filepath']) as img:
                    phash = str(imagehash.phash(img))

                if phash in phashes:
                    duplicates.append({
                        'original': phashes[phash],
                        'duplicate': row['filepath'],
                        'phash': phash,
                        'similarity': 'perceptual'
                    })
                else:
                    phashes[phash] = row['filepath']
            except Exception as e:
                pass

        return duplicates

# Check for exact duplicates using file hash
import hashlib
duplicates = find_duplicates(df_raw, method='hash')

print(f"\nFound {len(duplicates)} exact duplicate files")
if duplicates:
    print("\nFirst 5 duplicates:")
    for dup in duplicates[:5]:
        print(f"  Original: {os.path.basename(dup['original'])}")
        print(f"  Duplicate: {os.path.basename(dup['duplicate'])}")
        print()

# Remove duplicates (keep first occurrence)
if duplicates:
    duplicate_paths = set([d['duplicate'] for d in duplicates])
    df_cleaned = df_raw[~df_raw['filepath'].isin(duplicate_paths)].reset_index(drop=True)
    print(f"Removed {len(duplicate_paths)} duplicate images")
    print(f"Cleaned dataset size: {len(df_cleaned)} (was {len(df_raw)})")
else:
    df_cleaned = df_raw.copy()
    print("No duplicates found. Dataset remains at {} images".format(len(df_cleaned)))

# ------------------------------------------------------------------------------
# 4.2 Corrupted Image Detection
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("4.2 CORRUPTED IMAGE DETECTION")
print("="*80)

def detect_corrupted_images(df):
    """
    Detect images that cannot be opened or have critical issues.
    """
    corrupted = []
    valid = []

    print("Checking for corrupted images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            with Image.open(row['filepath']) as img:
                # Try to load and verify
                img.verify()

            # Re-open for full check (verify closes the file)
            with Image.open(row['filepath']) as img:
                # Check basic properties
                if img.size[0] == 0 or img.size[1] == 0:
                    corrupted.append({'filepath': row['filepath'], 'reason': 'zero_dimension'})
                elif img.mode not in ['RGB', 'RGBA', 'L']:
                    corrupted.append({'filepath': row['filepath'], 'reason': f'unsupported_mode_{img.mode}'})
                else:
                    valid.append(row)
        except Exception as e:
            corrupted.append({'filepath': row['filepath'], 'reason': str(e)})

    return corrupted, pd.DataFrame(valid)

corrupted_files, df_valid = detect_corrupted_images(df_cleaned)

print(f"\nFound {len(corrupted_files)} corrupted/invalid images")
if corrupted_files:
    print("\nSample corrupted files:")
    for corr in corrupted_files[:5]:
        print(f"  {os.path.basename(corr['filepath'])}: {corr['reason']}")

print(f"\nValid images after cleaning: {len(df_valid)}")

# ------------------------------------------------------------------------------
# 4.3 Quality Metrics Summary
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("4.3 DATA QUALITY SUMMARY")
print("="*80)

quality_report = {
    'original_count': len(df_raw),
    'duplicates_removed': len(df_raw) - len(df_cleaned),
    'corrupted_removed': len(corrupted_files),
    'final_count': len(df_valid),
    'retention_rate': len(df_valid) / len(df_raw) * 100,
    'classes': df_valid['class_name'].nunique(),
    'crops': df_valid['crop_type'].nunique(),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

print("\n--- Quality Report ---")
for key, value in quality_report.items():
    print(f"  {key}: {value}")

# Save quality report
with open(os.path.join(PROCESSED_DIR, 'quality_report.json'), 'w') as f:
    json.dump(quality_report, f, indent=2)

print(f"\nQuality report saved to: {os.path.join(PROCESSED_DIR, 'quality_report.json')}")

# Update df_raw to cleaned version for further processing
df_raw = df_valid.reset_index(drop=True)

# ==============================================================================
# 5. FEATURE ENGINEERING
# ==============================================================================
"""
## 5. Feature Engineering

This section implements the preprocessing pipeline as specified in the 
project documentation (Section 3.3 and Table 3.3, 3.4).

**Preprocessing Pipeline:**
1. **P1-P5:** Loading, verification, resizing, color space, normalization
2. **Augmentation:** Rotation, shift, flip, zoom, brightness (Table 3.4)
3. **Stratified Splitting:** 70/15/15 split preserving class distribution
4. **Label Encoding:** One-hot encoding for 14 classes

**Target Specifications:**
- Input size: 224×224×3
- Pixel values: [0, 1] (normalized)
- Augmentation: Applied only to training set
- Class weights: Computed for imbalance handling
"""

# ------------------------------------------------------------------------------
# 5.1 Image Preprocessing Functions
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("5.1 IMAGE PREPROCESSING PIPELINE")
print("="*80)

# Configuration from project documentation
IMG_SIZE = (224, 224)  # As per Table 3.5
BATCH_SIZE = 32

class ImagePreprocessor:
    """
    Comprehensive image preprocessing pipeline for crop disease classification.
    Implements stages P1-P5 from Table 3.3 of the project documentation.
    """

    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize

    def load_image(self, filepath):
        """P1: Image Loading with format standardization"""
        try:
            img = Image.open(filepath)
            return img
        except Exception as e:
            raise ValueError(f"Cannot load image {filepath}: {e}")

    def verify_dimensions(self, img, min_size=224):
        """P2: Dimension Verification"""
        width, height = img.size
        if width < min_size or height < min_size:
            raise ValueError(f"Image too small: {width}x{height}, min required: {min_size}x{min_size}")
        return True

    def resize_image(self, img):
        """P3: Resizing with bicubic interpolation"""
        return img.resize(self.target_size, Image.BICUBIC)

    def standardize_color(self, img):
        """P4: Color Space standardization (RGB)"""
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                # Create white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert('RGB')
        return img

    def normalize_image(self, img_array):
        """P5: Normalization to [0, 1] range"""
        if self.normalize:
            return img_array.astype(np.float32) / 255.0
        return img_array

    def preprocess(self, filepath):
        """Full preprocessing pipeline"""
        # Load
        img = self.load_image(filepath)

        # Verify
        self.verify_dimensions(img)

        # Resize
        img = self.resize_image(img)

        # Color standardization
        img = self.standardize_color(img)

        # Convert to array
        img_array = np.array(img)

        # Normalize
        img_array = self.normalize_image(img_array)

        return img_array

# Initialize preprocessor
preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)

# Test preprocessing on sample
print("Testing preprocessing pipeline on sample images...")
sample_indices = np.random.choice(len(df_raw), 5, replace=False)
for idx in sample_indices:
    row = df_raw.iloc[idx]
    try:
        processed = preprocessor.preprocess(row['filepath'])
        print(f"  ✓ {row['filename']}: {processed.shape}, range [{processed.min():.3f}, {processed.max():.3f}]")
    except Exception as e:
        print(f"  ✗ {row['filename']}: {e}")

print("\nPreprocessing pipeline validated!")

# ------------------------------------------------------------------------------
# 5.2 Data Augmentation Strategy
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("5.2 DATA AUGMENTATION STRATEGY")
print("="*80)

"""
Augmentation parameters from Table 3.4 of documentation:
- Rotation: ±20 degrees (prob 0.3)
- Width Shift: ±20% (prob 0.3)
- Height Shift: ±20% (prob 0.3)
- Horizontal Flip: 0.5 probability
- Zoom: [0.8, 1.2] (prob 0.3)
- Brightness: [0.8, 1.2] (prob 0.3)
- Fill Mode: Nearest
"""

# Create augmentation configuration
augmentation_config = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': [0.8, 1.2],
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest',
    'rescale': 1.0/255.0  # Normalization included
}

print("Augmentation Configuration (from Table 3.4):")
for key, value in augmentation_config.items():
    print(f"  {key}: {value}")

# Create Keras ImageDataGenerator for training
train_datagen = ImageDataGenerator(
    rotation_range=augmentation_config['rotation_range'],
    width_shift_range=augmentation_config['width_shift_range'],
    height_shift_range=augmentation_config['height_shift_range'],
    horizontal_flip=augmentation_config['horizontal_flip'],
    zoom_range=augmentation_config['zoom_range'],
    brightness_range=augmentation_config['brightness_range'],
    fill_mode=augmentation_config['fill_mode'],
    rescale=augmentation_config['rescale']
)

# Validation/Test: Only normalization
val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Visualize augmentation effects
print("\nGenerating augmentation visualization...")

def visualize_augmentation(filepath, datagen, n_augmented=5):
    """Visualize augmentation effects on a single image"""
    # Load and preprocess base image
    img = Image.open(filepath).convert('RGB').resize((224, 224))
    img_array = np.array(img)

    fig, axes = plt.subplots(1, n_augmented + 1, figsize=(18, 3))

    # Original
    axes[0].imshow(img_array)
    axes[0].set_title('Original', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Augmented versions
    img_batch = np.expand_dims(img_array, axis=0)
    aug_iter = datagen.flow(img_batch, batch_size=1)

    for i in range(n_augmented):
        aug_img = next(aug_iter)[0].astype(np.uint8)
        axes[i+1].imshow(aug_img)
        axes[i+1].set_title(f'Augmented {i+1}', fontsize=10)
        axes[i+1].axis('off')

    plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=200, bbox_inches='tight')
    plt.show()

# Show augmentation on a sample
sample_file = df_raw.iloc[0]['filepath']
visualize_augmentation(sample_file, train_datagen, n_augmented=5)
print("\nVisualization saved: augmentation_examples.png")

# ------------------------------------------------------------------------------
# 5.3 Stratified Data Splitting
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("5.3 STRATIFIED DATA SPLITTING")
print("="*80)

"""
According to Section 3.3 and Table 3.3, we implement stratified splitting:
- Train: 70%
- Validation: 15%
- Test: 15%

Stratification ensures class distribution is preserved across all splits,
critical for handling class imbalance (Section 3.3).
"""

# Encode labels for stratification
label_encoder = LabelEncoder()
df_raw['class_encoded'] = label_encoder.fit_transform(df_raw['class_name'])

# Get class names for reference
class_names = label_encoder.classes_
print(f"Classes ({len(class_names)}):")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# First split: Separate test set (15%)
print("\nPerforming stratified split...")
X_temp, X_test, y_temp, y_test = train_test_split(
    df_raw['filepath'].values,
    df_raw['class_encoded'].values,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=df_raw['class_encoded'].values
)

# Second split: Separate train and validation (70% / 15% of total = 82.35% / 17.65% of temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15
    random_state=RANDOM_SEED,
    stratify=y_temp
)

# Create split dataframes
def create_split_df(filepaths, labels, split_name):
    df = pd.DataFrame({
        'filepath': filepaths,
        'class_encoded': labels,
        'split': split_name
    })
    # Add metadata from original df
    df = df.merge(df_raw[['filepath', 'class_name', 'crop_type', 'health_status', 'disease_name']], 
                  on='filepath', how='left')
    return df

df_train = create_split_df(X_train, y_train, 'train')
df_val = create_split_df(X_val, y_val, 'val')
df_test = create_split_df(X_test, y_test, 'test')

# Combine all splits
df_split = pd.concat([df_train, df_val, df_test], ignore_index=True)

print("\n--- Split Summary ---")
print(f"Training: {len(df_train)} ({len(df_train)/len(df_raw)*100:.1f}%)")
print(f"Validation: {len(df_val)} ({len(df_val)/len(df_raw)*100:.1f}%)")
print(f"Test: {len(df_test)} ({len(df_test)/len(df_raw)*100:.1f}%)")
print(f"Total: {len(df_split)}")

# Verify stratification
print("\n--- Class Distribution Across Splits ---")
strat_check = pd.crosstab(df_split['class_name'], df_split['split'], normalize='columns') * 100
print(strat_check.round(1))

# Visualization of stratification
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Absolute counts
split_counts = df_split.groupby(['split', 'class_name']).size().unstack(fill_value=0)
split_counts.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab20')
axes[0].set_title('Absolute Counts by Split and Class', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Split')
axes[0].set_ylabel('Number of Images')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Percentage distribution
split_pct = df_split.groupby(['split', 'class_name']).size().groupby(level=0).apply(lambda x: x/x.sum()*100).unstack(fill_value=0)
split_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab20')
axes[1].set_title('Percentage Distribution by Split', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Split')
axes[1].set_ylabel('Percentage (%)')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('stratified_split_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nVisualization saved: stratified_split_distribution.png")

# ------------------------------------------------------------------------------
# 5.4 Class Weights Calculation
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("5.4 CLASS WEIGHTS CALCULATION")
print("="*80)

"""
Class weights are computed to handle imbalance during training (Section 3.6).
Formula: weight = total_samples / (n_classes * class_count)
"""

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df_train['class_encoded']),
    y=df_train['class_encoded']
)

class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Class Weights (for training):")
for i, (cls_idx, weight) in enumerate(class_weight_dict.items()):
    class_name = class_names[cls_idx]
    count = (df_train['class_encoded'] == cls_idx).sum()
    print(f"  Class {cls_idx} ({class_name[:30]:<30}): weight={weight:.4f}, count={count}")

# Save class weights
with open(os.path.join(PROCESSED_DIR, 'class_weights.json'), 'w') as f:
    json.dump(class_weight_dict, f, indent=2)

print(f"\nClass weights saved to: {os.path.join(PROCESSED_DIR, 'class_weights.json')}")

# ------------------------------------------------------------------------------
# 5.5 Metadata and Label Encoding
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("5.5 METADATA & LABEL ENCODING")
print("="*80)

# Create comprehensive metadata
metadata = {
    'dataset_info': {
        'name': 'New Plant Diseases Dataset (Processed)',
        'original_source': 'Kaggle - Ahmed (2019)',
        'total_images': len(df_split),
        'n_classes': len(class_names),
        'n_crops': df_split['crop_type'].nunique(),
        'class_names': list(class_names),
        'crop_types': list(df_split['crop_type'].unique())
    },
    'preprocessing': {
        'image_size': IMG_SIZE,
        'normalization': 'pixel/255.0',
        'color_mode': 'RGB',
        'interpolation': 'bicubic'
    },
    'augmentation': augmentation_config,
    'split_info': {
        'train_count': len(df_train),
        'val_count': len(df_val),
        'test_count': len(df_test),
        'train_pct': len(df_train) / len(df_split) * 100,
        'val_pct': len(df_val) / len(df_split) * 100,
        'test_pct': len(df_test) / len(df_split) * 100,
        'stratified': True,
        'random_seed': RANDOM_SEED
    },
    'class_mapping': {i: name for i, name in enumerate(class_names)},
    'derived_attributes': {
        'crop_type_mapping': df_split.groupby('class_name')['crop_type'].first().to_dict(),
        'health_status_mapping': df_split.groupby('class_name')['health_status'].first().to_dict()
    }
}

# Save metadata
with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved!")
print("\n--- Dataset Metadata Summary ---")
print(f"Total Images: {metadata['dataset_info']['total_images']}")
print(f"Classes: {metadata['dataset_info']['n_classes']}")
print(f"Crops: {', '.join(metadata['dataset_info']['crop_types'])}")
print(f"Split: {metadata['split_info']['train_pct']:.1f}% / {metadata['split_info']['val_pct']:.1f}% / {metadata['split_info']['test_pct']:.1f}%")

# ==============================================================================
# 6. DATA EXPORT & PIPELINE SERIALIZATION
# ==============================================================================
"""
## 6. Data Export & Pipeline Serialization

Export processed data for model training:
1. CSV manifests for each split
2. Serialized preprocessing pipeline
3. Directory structure for Keras ImageDataGenerator
4. Numpy arrays (optional, for faster loading)
"""

print("\n" + "="*80)
print("6. DATA EXPORT & PIPELINE SERIALIZATION")
print("="*80)

# 6.1 Save CSV manifests
print("\nSaving CSV manifests...")
df_train.to_csv(os.path.join(PROCESSED_DIR, 'train_manifest.csv'), index=False)
df_val.to_csv(os.path.join(PROCESSED_DIR, 'val_manifest.csv'), index=False)
df_test.to_csv(os.path.join(PROCESSED_DIR, 'test_manifest.csv'), index=False)
df_split.to_csv(os.path.join(PROCESSED_DIR, 'full_manifest.csv'), index=False)

print(f"  ✓ train_manifest.csv ({len(df_train)} records)")
print(f"  ✓ val_manifest.csv ({len(df_val)} records)")
print(f"  ✓ test_manifest.csv ({len(df_test)} records)")
print(f"  ✓ full_manifest.csv ({len(df_split)} records)")

# 6.2 Save label encoders
print("\nSaving encoders...")
with open(os.path.join(PROCESSED_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

# One-hot encoding example (for reference)
y_train_onehot = to_categorical(df_train['class_encoded'], num_classes=len(class_names))
y_val_onehot = to_categorical(df_val['class_encoded'], num_classes=len(class_names))
y_test_onehot = to_categorical(df_test['class_encoded'], num_classes=len(class_names))

np.save(os.path.join(PROCESSED_DIR, 'y_train_onehot.npy'), y_train_onehot)
np.save(os.path.join(PROCESSED_DIR, 'y_val_onehot.npy'), y_val_onehot)
np.save(os.path.join(PROCESSED_DIR, 'y_test_onehot.npy'), y_test_onehot)

print(f"  ✓ label_encoder.pkl")
print(f"  ✓ y_train_onehot.npy (shape: {y_train_onehot.shape})")
print(f"  ✓ y_val_onehot.npy (shape: {y_val_onehot.shape})")
print(f"  ✓ y_test_onehot.npy (shape: {y_test_onehot.shape})")

# 6.3 Create organized directory structure for ImageDataGenerator
print("\nCreating organized directory structure...")

def create_image_folders(df, target_dir):
    """Create class folders and copy/symlink images"""
    os.makedirs(target_dir, exist_ok=True)

    for class_name in df['class_name'].unique():
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Get images for this class
        class_images = df[df['class_name'] == class_name]['filepath'].tolist()

        # Create symlinks (or copy if symlinks not supported)
        for src_path in class_images:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(class_dir, filename)

            if not os.path.exists(dst_path):
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except (OSError, NotImplementedError):
                    # Fallback to copy if symlinks not supported
                    shutil.copy2(src_path, dst_path)

# Create folder structure for each split
create_image_folders(df_train, os.path.join(SPLIT_DIR, 'train'))
create_image_folders(df_val, os.path.join(SPLIT_DIR, 'val'))
create_image_folders(df_test, os.path.join(SPLIT_DIR, 'test'))

print(f"  ✓ {SPLIT_DIR}/train/ ({len(df_train)} images)")
print(f"  ✓ {SPLIT_DIR}/val/ ({len(df_val)} images)")
print(f"  ✓ {SPLIT_DIR}/test/ ({len(df_test)} images)")

# 6.4 Save preprocessing pipeline
print("\nSaving preprocessing pipeline...")
pipeline_config = {
    'preprocessor_params': {
        'target_size': IMG_SIZE,
        'normalize': True
    },
    'augmentation_config': augmentation_config,
    'class_weight_dict': class_weight_dict,
    'random_seed': RANDOM_SEED
}

with open(os.path.join(PROCESSED_DIR, 'pipeline_config.pkl'), 'wb') as f:
    pickle.dump(pipeline_config, f)

print(f"  ✓ pipeline_config.pkl")

# ==============================================================================
# 7. SUMMARY & QUALITY REPORT
# ==============================================================================
print("\n" + "="*80)
print("7. FINAL SUMMARY & QUALITY REPORT")
print("="*80)

# Generate comprehensive summary
summary_report = f"""
================================================================================
NEURALNEST CROP DISEASE CLASSIFICATION - DATA PREPARATION REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET OVERVIEW
--------------------------------------------------------------------------------
Original Dataset:      New Plant Diseases Dataset (Ahmed, 2019)
Total Images:          {len(df_raw):,}
Number of Classes:     {len(class_names)}
Number of Crops:       {df_raw['crop_type'].nunique()} (Corn, Potato, Wheat)

2. DATA CLEANING RESULTS
--------------------------------------------------------------------------------
Duplicates Removed:    {quality_report['duplicates_removed']}
Corrupted Removed:     {quality_report['corrupted_removed']}
Final Clean Images:    {quality_report['final_count']:,}
Retention Rate:        {quality_report['retention_rate']:.2f}%

3. CLASS DISTRIBUTION
--------------------------------------------------------------------------------
Imbalance Ratio:       {imbalance_ratio:.2f} (Max/Min)
Balanced Status:       {'Yes' if imbalance_ratio < 1.5 else 'Moderate' if imbalance_ratio < 3 else 'High Imbalance - Class weights applied'}

Class Breakdown:
"""

for i, (class_name, count) in enumerate(class_counts.items()):
    crop = df_raw[df_raw['class_name'] == class_name]['crop_type'].iloc[0]
    health = df_raw[df_raw['class_name'] == class_name]['health_status'].iloc[0]
    summary_report += f"  {i+1:2d}. {class_name[:40]:<40} | {count:4d} | {crop:<6} | {health}
"

summary_report += f"""
4. DATA SPLITTING (Stratified)
--------------------------------------------------------------------------------
Training Set:          {len(df_train):5,} images ({len(df_train)/len(df_split)*100:5.1f}%)
Validation Set:        {len(df_val):5,} images ({len(df_val)/len(df_split)*100:5.1f}%)
Test Set:              {len(df_test):5,} images ({len(df_test)/len(df_split)*100:5.1f}%)
Random Seed:           {RANDOM_SEED}

5. PREPROCESSING PIPELINE
--------------------------------------------------------------------------------
Target Size:           {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels
Color Mode:            RGB
Normalization:         Pixel/255.0 → [0,1]
Interpolation:         Bicubic

6. AUGMENTATION (Training Set Only)
--------------------------------------------------------------------------------
Rotation Range:        ±{augmentation_config['rotation_range']}°
Width Shift:           ±{augmentation_config['width_shift_range']*100:.0f}%
Height Shift:          ±{augmentation_config['height_shift_range']*100:.0f}%
Horizontal Flip:       Yes
Zoom Range:            [{augmentation_config['zoom_range'][0]}, {augmentation_config['zoom_range'][1]}]
Brightness Range:      [{augmentation_config['brightness_range'][0]}, {augmentation_config['brightness_range'][1]}]

7. EXPORTED FILES
--------------------------------------------------------------------------------
Location:              {PROCESSED_DIR}/
  - train_manifest.csv ({len(df_train)} records)
  - val_manifest.csv ({len(df_val)} records)
  - test_manifest.csv ({len(df_test)} records)
  - metadata.json (full dataset metadata)
  - class_weights.json (for imbalanced training)
  - label_encoder.pkl (sklearn encoder)
  - pipeline_config.pkl (full pipeline config)
  - quality_report.json (cleaning log)

Image Folders:         {SPLIT_DIR}/
  - train/ (organized by class)
  - val/ (organized by class)
  - test/ (organized by class)

8. NEXT STEPS
--------------------------------------------------------------------------------
✓ Data is ready for model training
✓ Use ImageDataGenerator with provided augmentation config
✓ Apply class weights during model training for imbalance handling
✓ All splits are stratified and reproducible (seed: {RANDOM_SEED})

================================================================================
END OF REPORT
================================================================================
"""

print(summary_report)

# Save summary report
with open(os.path.join(PROCESSED_DIR, 'PREPARATION_REPORT.txt'), 'w') as f:
    f.write(summary_report)

print(f"\nFull report saved to: {os.path.join(PROCESSED_DIR, 'PREPARATION_REPORT.txt')}")

# Final checklist
print("\n" + "="*80)
print("DELIVERABLES CHECKLIST")
print("="*80)
deliverables = [
    ("EDA Notebook", "This notebook with all visualizations"),
    ("CSV Manifests", "train_manifest.csv, val_manifest.csv, test_manifest.csv"),
    ("Metadata", "metadata.json with full dataset info"),
    ("Preprocessing Pipeline", "pipeline_config.pkl for reproduction"),
    ("Class Weights", "class_weights.json for training"),
    ("Organized Images", "Stratified split folder structure"),
    ("Quality Report", "quality_report.json and PREPARATION_REPORT.txt"),
    ("Visualizations", "eda_*.png and augmentation_examples.png")
]

for i, (item, desc) in enumerate(deliverables, 1):
    print(f"{i}. ✓ {item}")
    print(f"   └─ {desc}")

print("="*80)
print("DATA PREPARATION COMPLETE - READY FOR MODEL TRAINING")
print("="*80)
