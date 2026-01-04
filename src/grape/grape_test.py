# Test GRAPE model
from .grape_preprocessing import get_grape_generators
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

BASE_DIR = "dataset/image data"
MODEL_PATH = os.path.join("grape", "grape_model.h5")

# Load test generator
_, _, test_gen = get_grape_generators(BASE_DIR)

# Load model
model = load_model(MODEL_PATH)

# Predict all test images
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# Evaluate
print("Classification Report:")
print(classification_report(
    y_true, y_pred, target_names=list(test_gen.class_indices.keys())
))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
