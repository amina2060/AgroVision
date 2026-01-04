# Data preprocessing for CASSAVA
import os
from src.common.preprocessing.image_utils import create_data_generator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_cassava_generators(base_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Returns training, validation, and test generators for Cassava crop.
    """
    train_dir = os.path.join(base_dir, "train/cassava")
    val_dir = os.path.join(base_dir, "validation/cassava")
    test_dir = os.path.join(base_dir, "test/cassava")

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
