import cv2
from .rice_segmentation_utils import read_and_gray, threshold_mask, overlay_mask

def segment_rice_leaf(image_path):
    img, gray = read_and_gray(image_path)
    mask = threshold_mask(gray)
    overlay = overlay_mask(img, mask)
    return mask, overlay
