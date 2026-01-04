# src/main.py
# Main pipeline
# 1. Predict crop
# 2. Route to crop-specific disease model

from tkinter import Tk, filedialog
from src.common.crop_identifier import identify_crop

# -------------------------------
# Step 0: Select image using file dialog
# -------------------------------
root = Tk()
root.withdraw()  # Hide tkinter root window

# Make the file dialog appear on top
root.attributes('-topmost', True)

image_path = filedialog.askopenfilename(
    title="Select Leaf Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected. Exiting.")
    exit()

# -------------------------------
# Step 1: Identify crop with confidence
# -------------------------------
crop_name, confidence = identify_crop(image_path)
print(f"Detected crop: {crop_name} (confidence: {confidence:.2f})")

# -------------------------------
# Step 2: Check confidence threshold
# -------------------------------
if crop_name == "unknown":
    print("Crop prediction confidence too low. Cannot proceed with disease prediction.")
    disease = None
else:
    # -------------------------------
    # Step 3: Route to crop-specific predictor
    # -------------------------------
    if crop_name == "apple":
        from src.apple.apple_predict import predict_apple
        from src.apple.apple_classes import CLASSES
        disease_index = predict_apple(image_path)

    elif crop_name == "tomato":
        from src.tomato.tomato_predict import predict_tomato
        from src.tomato.tomato_classes import CLASSES
        disease_index = predict_tomato(image_path)

    elif crop_name == "corn":
        from src.corn.corn_predict import predict_corn
        from src.corn.corn_classes import CLASSES
        disease_index = predict_corn(image_path)

    elif crop_name == "rice":
        from src.rice.rice_predict import predict_rice
        from src.rice.rice_classes import CLASSES
        disease_index = predict_rice(image_path)

    elif crop_name == "grape":
        from src.grape.grape_predict import predict_grape
        from src.grape.grape_classes import CLASSES
        disease_index = predict_grape(image_path)

    elif crop_name == "cassava":
        from src.cassava.cassava_predict import predict_cassava
        from src.cassava.cassava_classes import CLASSES
        disease_index = predict_cassava(image_path)

    else:
        print(f"No predictor available for crop: {crop_name}")
        disease = None

    # -------------------------------
    # Step 4: Print predicted disease
    # -------------------------------
    if disease_index is not None:
        try:
            disease = CLASSES[disease_index]
            print(f"Predicted disease: {disease}")
        except Exception:
            print("Disease prediction could not be mapped to a class name.")
    else:
        print("Disease prediction could not be made.")
# -------------------------------
# Step 5: Segment diseased regions (generic for any crop)
# -------------------------------
import cv2
import matplotlib.pyplot as plt
from src.common.analysis.severity import calculate_severity, infection_severity
from src.common.visualization.visualize import show_overlay_with_severity

segmentation_map = {
    "apple": ("src.apple.segmentation.apple_segmentation", "segment_apple_leaf"),
    "tomato": ("src.tomato.segmentation.tomato_segmentation", "segment_tomato_leaf"),
    "corn": ("src.corn.segmentation.corn_segmentation", "segment_corn_leaf"),
    "rice": ("src.rice.segmentation.rice_segmentation", "segment_rice_leaf"),
    "grape": ("src.grape.segmentation.grape_segmentation", "segment_grape_leaf"),
    "cassava": ("src.cassava.segmentation.cassava_segmentation", "segment_cassava_leaf")
}

if crop_name in segmentation_map:
    module_path, func_name = segmentation_map[crop_name]
    seg_module = __import__(module_path, fromlist=[func_name])
    seg_func = getattr(seg_module, func_name)
    mask, overlay = seg_func(image_path)

    # -------------------------------
    # Show the original, mask, and overlay side by side
    # -------------------------------
    img = cv2.imread(image_path)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

    # -------------------------------
    # Step 6: Calculate severity
    # -------------------------------
    severity_percent = calculate_severity(mask)
    severity_level = infection_severity(severity_percent)
    
    print(f"Disease severity: {severity_percent:.2f}% ({severity_level})")

    # -------------------------------
    # Optional: show overlay with severity
    # -------------------------------
    show_overlay_with_severity(overlay, severity_percent)

else:
    print(f"No segmentation available for crop: {crop_name}")
