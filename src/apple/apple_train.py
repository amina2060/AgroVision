# apple/apple_train.py
from .apple_preprocessing import get_apple_generators
from src.common.models.model_base import build_model

BASE_DIR = "dataset/image data"

train_gen, val_gen, test_gen = get_apple_generators(BASE_DIR)
num_classes = len(train_gen.class_indices)
model = build_model(num_classes=num_classes)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=1
)

model.save("apple/apple_model.h5")
print("Apple model trained and saved!")
