# src/data_preprocessing.py
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SELECTED_CROPS = ['tomato', 'apple', 'rice', 'cassava', 'grape', 'corn']

def extract_leaf_features(image):
    """Extract shape/dimension features from a leaf image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(12, dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / (h + 1e-5)
    rect_area = w * h
    extent = float(area) / (rect_area + 1e-5)
    hull = cv2.convexHull(cnt)
    solidity = float(area) / (cv2.contourArea(hull) + 1e-5)
    moments = cv2.HuMoments(cv2.moments(cnt)).flatten()
    features = [area, perimeter, aspect_ratio, extent, solidity] + moments.tolist()
    return np.array(features, dtype=np.float32)

def get_data_generators(train_dir, val_dir):
    # Only use selected crops
    def filter_selected_crops(path, class_mode='categorical'):
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            classes=SELECTED_CROPS,
            class_mode=class_mode
        )
        return generator

    train_generator = filter_selected_crops(train_dir)
    val_generator = filter_selected_crops(val_dir)
    return train_generator, val_generator
