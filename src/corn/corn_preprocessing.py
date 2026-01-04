# src/corn/corn_preprocessing.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.common.preprocessing.image_utils import create_data_generator  # optional common generator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_corn_generators(base_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Returns training, validation, and test generators for Corn crop.
    Works exactly like Apple generators.
    """
    train_dir = os.path.join(base_dir, "train/corn (maize)")
    val_dir   = os.path.join(base_dir, "validation/corn (maize)")
    test_dir  = os.path.join(base_dir, "test/corn (maize)")

    # Use generic data generator if available
    train_datagen = create_data_generator(augment=True)
    val_test_datagen = create_data_generator(augment=False)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen
