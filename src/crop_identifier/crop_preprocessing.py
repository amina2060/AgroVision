# src/crop_identifier/crop_preprocessing.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.common.preprocessing.image_utils import create_data_generator
from .crop_classes import FOLDER_TO_CLASS

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Only use the folders listed in FOLDER_TO_CLASS
ALLOWED_FOLDERS = list(FOLDER_TO_CLASS.keys())

def get_crop_generators(base_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Returns training, validation, and test generators for crop classification.
    Only uses the 6 crops defined in FOLDER_TO_CLASS.
    """
    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "validation")
    test_dir  = os.path.join(base_dir, "test")

    # Filter to include only allowed folders
    def filter_dirs(parent_dir):
        return [d for d in os.listdir(parent_dir) if d in ALLOWED_FOLDERS]

    # Training generator
    train_datagen = create_data_generator(augment=True)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=filter_dirs(train_dir)  # only selected folders
    )

    # Validation generator
    val_datagen = create_data_generator(augment=False)
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=filter_dirs(val_dir)
    )

    # Test generator
    test_datagen = create_data_generator(augment=False)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=filter_dirs(test_dir)
    )

    return train_gen, val_gen, test_gen
