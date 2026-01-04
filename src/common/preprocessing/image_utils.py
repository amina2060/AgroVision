# src/common/preprocessing/image_utils.py

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

def create_data_generator(rescale=1./255, augment=False):
    """
    Returns a Keras ImageDataGenerator.
    
    Args:
        rescale (float): Rescaling factor for pixel values.
        augment (bool): Whether to apply data augmentation.
    
    Returns:
        ImageDataGenerator object
    """
    if augment:
        return ImageDataGenerator(
            rescale=rescale,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1
        )
    else:
        return ImageDataGenerator(rescale=rescale)


def get_data_generator_for_crop(crop_name, base_path="dataset/image data", augment=False):
    """
    Returns train, validation, and test generators for any crop dynamically.
    
    Args:
        crop_name (str): Name of the crop folder.
        base_path (str): Root dataset folder.
        augment (bool): Whether to apply augmentation on training set.
    
    Returns:
        train_gen, val_gen, test_gen
    """
    train_dir = os.path.join(base_path, "train", crop_name)
    val_dir = os.path.join(base_path, "validation", crop_name)
    test_dir = os.path.join(base_path, "test", crop_name)

    train_gen = create_data_generator(augment=augment).flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_gen = create_data_generator(augment=False).flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_gen = create_data_generator(augment=False).flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def preprocess_single_image(img_path):
    """
    Preprocessing for a single image path for prediction.
    
    Args:
        img_path (str): Path to image file.
    
    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
