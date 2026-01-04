# Segment diseased areas in APPLE leaves
# apple/segmentation/apple_segmentation.py

import cv2
import numpy as np
from .apple_segmentation_utils import read_and_gray, threshold_mask, overlay_mask

def segment_apple_leaf(image_path):
    """
    Returns a mask highlighting diseased regions on apple leaf.
    """
    img, gray = read_and_gray(image_path)
    mask = threshold_mask(gray)
    overlay = overlay_mask(img, mask)
    return mask, overlay
