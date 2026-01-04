# src/crop_identifier/crop_predict.py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from src.crop_identifier.crop_classes import CLASSES

MODEL_PATH = "src/crop_identifier/crop_model.h5"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.8  # 80% threshold

model = load_model(MODEL_PATH)

def predict_crop(image_path):
    """
    Predicts crop type from leaf image with confidence.
    Returns:
        (label, confidence) -> (str, float)
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    max_confidence = np.max(preds)
    crop_index = np.argmax(preds)

    if max_confidence < CONFIDENCE_THRESHOLD:
        return "unknown", max_confidence
    else:
        return CLASSES[crop_index], max_confidence
