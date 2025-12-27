import os
from src.predict import load_leaf_model, predict_leaf  # your predict.py functions

# =========================
# Detect project root dynamically
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # src folder
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))  # go to LeafDiseaseProject

# Path to test dataset
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "image data", "test")

# Classes and their subclasses
classes_to_test = {
    "apple": ["apple scab", "black rot", "cedar apple rust", "healthy"],
    "tomato": [
        "bacterial spot", "early blight", "healthy", "late blight", "leaf mold",
        "septoria leaf spot", "spider mites two-spotted spider mite",
        "target spot", "tomato mosaic virus", "tomato yellow leaf curl virus"
    ]
}

# =========================
# Load trained model
# =========================
print("Loading trained model...")
model = load_leaf_model()
print("Model loaded successfully.\n")

# =========================
# Testing loop
# =========================
total_images = 0
correct_predictions = 0

for main_class, subclasses in classes_to_test.items():
    for subclass in subclasses:
        subclass_path = os.path.join(TEST_DIR, main_class, subclass)
        if not os.path.isdir(subclass_path):
            print(f"⚠️  Folder not found: {subclass_path}")
            continue

        print(f"\nProcessing {main_class} -> {subclass}...")

        for img_file in os.listdir(subclass_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(subclass_path, img_file)
                pred_idx, confidence = predict_leaf(img_path, model)

                # Determine predicted class based on model output
                predicted_class = "apple" if pred_idx == 0 else "tomato"

                if predicted_class.lower() == main_class.lower():
                    correct_predictions += 1
                total_images += 1

                print(f"{img_file} -> Predicted: {predicted_class}, "
                      f"Actual: {main_class}/{subclass}, Confidence: {confidence:.2f}")

# =========================
# Final Accuracy
# =========================
if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100
    print(f"\n✅ Test completed. Total images: {total_images}, "
          f"Correct predictions: {correct_predictions}")
    print(f"Test Accuracy (Apple vs Tomato): {accuracy:.2f}%")
else:
    print("\n❌ No test images found. Check file paths and extensions.")
