# src/corn/corn_train.py
from .corn_preprocessing import get_corn_generators
from src.common.models.model_base import build_model

BASE_DIR = "dataset/image data"  # same as Apple

train_gen, val_gen, test_gen = get_corn_generators(BASE_DIR)
num_classes = len(train_gen.class_indices)

# Print class indices
print("Class indices:", train_gen.class_indices)

model = build_model(num_classes=num_classes)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=40,
    verbose=1
)

model.save("corn/corn_model.h5")
print("Corn model trained and saved!")
