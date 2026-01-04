# Helper functions for apple segmentation

import cv2
import numpy as np

def read_and_gray(image_path):
    """
    Read image and convert to grayscale.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def threshold_mask(gray_img, thresh_val=128):
    """
    Apply simple inverse binary threshold to create mask.
    """
    _, mask = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return mask

def overlay_mask(img, mask, alpha=0.3):
    """
    Overlay mask on original image.
    """
    overlay = cv2.addWeighted(img, 1 - alpha, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), alpha, 0)
    return overlay
