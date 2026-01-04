# apple/apple_preprocessing.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.common.preprocessing.image_utils import create_data_generator  # common generator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_apple_generators(base_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Returns training, validation, and test generators for Apple crop.
    """
    train_dir = os.path.join(base_dir, "train/apple")
    val_dir = os.path.join(base_dir, "validation/apple")
    test_dir = os.path.join(base_dir, "test/apple")

    # Use generic data generator from common utils
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
