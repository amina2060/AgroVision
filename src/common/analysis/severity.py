# src/common/analysis/severity.py
import cv2
import numpy as np

def calculate_severity(mask):
    """
    Calculates the % of leaf affected based on the mask.
    mask: binary image where diseased regions are white
    Returns: float percentage
    """
    # Ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    total_pixels = binary_mask.size
    diseased_pixels = cv2.countNonZero(binary_mask)
    severity_percent = (diseased_pixels / total_pixels) * 100
    return severity_percent

def infection_severity(infection_percent):
    """
    Converts % infection into categorical severity
    """
    if infection_percent < 20:
        return "Mild"
    elif infection_percent < 50:
        return "Moderate"
    else:
        return "Severe"
