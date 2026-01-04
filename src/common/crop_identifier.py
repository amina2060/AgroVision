# src/common/crop_identifier.py

from src.crop_identifier.crop_predict import predict_crop

def identify_crop(image_path):
    """
    Identifies crop from leaf image with confidence-based rejection.
    Returns:
        crop_name (str): one of CLASSES or 'unknown'
        confidence (float)
    """
    crop_name, confidence = predict_crop(image_path)
    return crop_name, confidence
