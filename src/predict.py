# src/predict.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

IMG_SIZE = (224, 224)

# =========================
# Safe absolute path to model
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model", "agrovision_model_M1.h5")
MODEL_PATH = os.path.abspath(MODEL_PATH)

def load_leaf_model():
    """Load the trained leaf disease model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return load_model(MODEL_PATH)

def predict_leaf(img_path, model):
    """Predict the class of a leaf image using the loaded model."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    return class_idx, confidence

# =========================
# Optional interactive testing
# =========================
if __name__ == "__main__":
    model = load_leaf_model()
    img_path = input("Enter image path: ").strip()
    idx, conf = predict_leaf(img_path, model)
    print(f"Predicted class: {idx}, Confidence: {conf:.2f}")
