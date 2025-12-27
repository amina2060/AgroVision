# src/train.py

import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from data_preprocessing import get_data_generators
from model import build_model

train_dir = r"..\dataset\image data\train"
val_dir   = r"..\dataset\image data\validation"

CHECKPOINT_DIR = r"..\checkpoints"
FINAL_MODEL_PATH = r"..\saved_model\agrovision_model.h5"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_gen, val_gen = get_data_generators(train_dir, val_dir)

num_classes = train_gen.num_classes
print("Classes detected:")
for k, v in train_gen.class_indices.items():
    print(v, ":", k)

model = build_model(num_classes)

checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "epoch_{epoch:02d}.h5"),
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[checkpoint_cb, earlystop_cb]
)

model.save(FINAL_MODEL_PATH)
print("✅ Training complete")
