import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path


# get path to root dataset 
dataset_dir = Path("data/Crop_Disease" )

crops=os.listdir("data/Crop_Disease")
class_names=[]
for crop in crops:
        crop_path = os.path.join(dataset_dir, crop)
        diseases = os.listdir(crop_path)

        for disease in diseases:
            disease_path = os.path.join(crop_path, disease)
            print(disease_path)

            class_name = f"{crop}_{disease}"
            class_names.append(class_name)
# 
full_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    label_mode='categorical',      
    image_size=(224, 224),          
    batch_size=32,
    shuffle=True
)

# get class names 
class_names = full_ds.class_names
print(class_names)


### EDA
## Count images per class
# select all jpg/jpeg images recursively
all_files = list(dataset_dir.glob("**/*.[jJ][pP][gG]"))
print(f"Total images found: {len(all_files)}")

# Get class labels from parent folder names
all_labels = [f.parent.name for f in all_files]
df_labels = pd.DataFrame(all_labels, columns=['Class'])

# Count images per class
class_counts = df_labels['Class'].value_counts().sort_values(ascending=True)
print(class_counts)

# Visualize
plt.figure(figsize=(12,9))
class_counts.plot(kind='barh')
plt.xlabel("Disease")
plt.ylabel("Number of images")
plt.title("Images per disease class")
plt.xticks(rotation=45)
plt.show()


from PIL import Image
import os

valid_images = []
for cls in full_ds.class_names:
    cls_path = os.path.join(dataset_dir, cls)
    for img_file in os.listdir(cls_path):
        path = os.path.join(cls_path, img_file)
        try:
            img = Image.open(path)
            img.verify()  # check if image is corrupted
            valid_images.append(path)
        except:
            print("Corrupt image:", path)


