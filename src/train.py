# src/train.py
from data_preprocessing import get_data_generators
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import os

# Paths (use raw strings to handle spaces correctly)
train_dir = r"..\dataset\image data\train"
val_dir   = r"..\dataset\image data\validation"
MODEL_PATH = r"..\saved_model\agrovision_model.h5"
EPOCH_FILE = r"..\saved_model\last_epoch.txt"
TOTAL_EPOCHS = 20  # total number of epochs you want to train

# Custom callback to save the last completed epoch
class EpochSaver(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(EPOCH_FILE, "w") as f:
            f.write(str(epoch + 1))  # save next epoch number

def train_model():
    # Check if paths exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")

    # Create data generators
    train_gen, val_gen = get_data_generators(train_dir, val_dir)
    print("Data generators ready. Starting training...")

    # Determine starting epoch
    initial_epoch = 0
    if os.path.exists(EPOCH_FILE):
        with open(EPOCH_FILE, "r") as f:
            initial_epoch = int(f.read())
        print(f"Resuming from epoch {initial_epoch + 1}")

    # Load existing model or start new
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Loaded existing model.")
        model.compile(
            optimizer=Adam(0.0001),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy']
        )
    else:
        model = build_model(num_classes=train_gen.num_classes)
        print("Starting a new model...")

    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    epoch_saver = EpochSaver()

    # Start or continue training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=TOTAL_EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint, early_stop, epoch_saver]
    )

    print("Training finished and model saved.")
    return model

if __name__ == "__main__":
    train_model()
