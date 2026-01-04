# Train rice model here
# rice/rice_train.py
from .rice_preprocessing import get_rice_generators
from src.common.models.model_base import build_model

BASE_DIR = "dataset/image data"

train_gen, val_gen, test_gen = get_rice_generators(BASE_DIR)

# 🔥 IMPORTANT: print class indices
print("Class indices:", train_gen.class_indices)

num_classes = len(train_gen.class_indices)

model = build_model(num_classes=num_classes)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    verbose=1
)

model.save("rice/rice_model.h5")
print("Rice model trained and saved!")
