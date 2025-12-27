# src/data_preprocessing.py

import os
import shutil
import tempfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_flat_directory(original_dir):
    """
    Converts:
    train/apple/apple scab/img.JPG
    INTO:
    temp/apple___apple scab/img.JPG
    """

    temp_dir = tempfile.mkdtemp()

    for crop in os.listdir(original_dir):
        crop_path = os.path.join(original_dir, crop)
        if not os.path.isdir(crop_path):
            continue

        for disease in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease)
            if not os.path.isdir(disease_path):
                continue

            class_name = f"{crop}___{disease}"
            class_dir = os.path.join(temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            for img in os.listdir(disease_path):
                if img.lower().endswith(".jpg"):
                    shutil.copy(
                        os.path.join(disease_path, img),
                        os.path.join(class_dir, img)
                    )

    return temp_dir


def get_data_generators(train_dir, val_dir, img_size=(224, 224), batch_size=32):

    flat_train = prepare_flat_directory(train_dir)
    flat_val   = prepare_flat_directory(val_dir)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        flat_train,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        flat_val,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen
