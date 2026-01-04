# Train tomato model here
# tomato/tomato_train.py

from .tomato_preprocessing import get_tomato_generators
from src.common.models.model_base import build_model

BASE_DIR = "dataset/image data"

train_gen, val_gen, test_gen = get_tomato_generators(BASE_DIR)

num_classes = len(train_gen.class_indices)
model = build_model(num_classes=num_classes)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=1
)

model.save("tomato/tomato_model.h5")
print("Tomato model trained and saved!")
