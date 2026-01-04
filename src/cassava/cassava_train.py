from src.cassava.cassava_preprocessing import get_cassava_generators
from src.common.models.model_base import build_model

BASE_DIR = "dataset/image data"

# Get generators
train_gen, val_gen, test_gen = get_cassava_generators(BASE_DIR)  # originally no augment

num_classes = len(train_gen.class_indices)
model = build_model(num_classes=num_classes)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=40,
    verbose=1
)

model.save("cassava/cassava_model.h5")
print("Cassava model trained and saved!")
