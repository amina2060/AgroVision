# Test RICE model
# rice/rice_test.py
from .rice_preprocessing import get_rice_generators
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

BASE_DIR = "dataset/image data"
MODEL_PATH = os.path.join("rice", "rice_model.h5")

_, _, test_gen = get_rice_generators(BASE_DIR)

model = load_model(MODEL_PATH)

preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(test_gen.class_indices.keys())
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
