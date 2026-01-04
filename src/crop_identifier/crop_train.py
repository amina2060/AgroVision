# src/crop_identifier/crop_train.py

from src.crop_identifier.crop_preprocessing import get_crop_generators
from src.common.models.model_base import build_model

BASE_DIR = "dataset/image data"

train_gen, val_gen, test_gen = get_crop_generators(BASE_DIR)
num_classes = len(train_gen.class_indices)

print("Class indices:", train_gen.class_indices)

model = build_model(num_classes=num_classes)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    verbose=1
)

model.save("src/crop_identifier/crop_model.h5")
print("Crop classifier trained and saved!")
