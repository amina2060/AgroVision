# src/predict.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_SIZE = (224, 224)
MODEL_PATH = '../saved_model/agrovision_model.h5'

def load_leaf_model():
    return load_model(MODEL_PATH)

def predict_leaf(img_path, model):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    return class_idx, confidence

if __name__ == "__main__":
    model = load_leaf_model()
    img_path = input("Enter image path: ")
    idx, conf = predict_leaf(img_path, model)
    print(f"Predicted class: {idx}, Confidence: {conf:.2f}")
