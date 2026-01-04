import cv2
from .cassava_segmentation_utils import read_and_gray, threshold_mask, overlay_mask

def segment_cassava_leaf(image_path):
    """
    Returns a mask highlighting diseased regions on cassava leaf.
    """
    img, gray = read_and_gray(image_path)
    mask = threshold_mask(gray)
    overlay = overlay_mask(img, mask)
    return mask, overlay

