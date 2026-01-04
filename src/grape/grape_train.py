# Train grape model here
from .grape_preprocessing import get_grape_generators
from src.common.models.model_base import build_model

BASE_DIR = "dataset/image data"

train_gen, val_gen, test_gen = get_grape_generators(BASE_DIR)
num_classes = len(train_gen.class_indices)
print("Class indices:", train_gen.class_indices)

model = build_model(num_classes=num_classes)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=40,  # you can adjust epochs
    verbose=1
)

model.save("grape/grape_model.h5")
print("Grape model trained and saved!")
