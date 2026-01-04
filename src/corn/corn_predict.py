# Predict disease for CORN
# corn/corn_predict.py
# Predict disease for CORN

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

MODEL_PATH = "corn/corn_model.h5"
IMG_SIZE = (224, 224)

# Load trained Corn model
model = load_model(MODEL_PATH)

def predict_corn(image_path, class_indices=None):
    """
    Predict disease for a single corn image.
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]

    if class_indices:
        # Map index to class name
        inv_map = {v: k for k, v in class_indices.items()}
        return inv_map[predicted_class]

    return predicted_class
