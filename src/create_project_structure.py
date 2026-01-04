import os

BASE_DIR = "src"

CROPS_CLASSES = {
    "apple": [
        "Apple Scab",
        "Black Rot",
        "Cedar Apple Rust",
        "Healthy"
    ],
    "tomato": [
        "Bacterial Spot",
        "Early Blight",
        "Late Blight",
        "Leaf Mold",
        "Septoria Leaf Spot",
        "Spider Mites (Two-spotted Spider Mite)",
        "Target Spot",
        "Tomato Mosaic Virus",
        "Tomato Yellow Leaf Curl Virus",
        "Healthy"
    ],
    "rice": [
        "Brown Spot",
        "Leaf Smut",
        "Leaf Blast",
        "Healthy"
    ],
    "corn": [
        "Cercospora Leaf Spot",
        "Common Rust",
        "Northern Leaf Blight",
        "Healthy"
    ],
    "grape": [
        "Black Rot",
        "Esca (Black Measles)",
        "Leaf Blight (Isariopsis Leaf Spot)",
        "Healthy"
    ],
    "cassava": [
        "Cassava Bacterial Blight",
        "Cassava Brown Streak Disease",
        "Cassava Green Mottle",
        "Healthy",
        "Cassava Mosaic Disease"
    ]
}

# Common modular files
COMMON_FILES = {
    "paths.py": """import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "image data")
""",
    "models/model_base.py": """from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224, 224)

def build_model(num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
""",
    "preprocessing/image_utils.py": """from tensorflow.keras.preprocessing import image
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
""",
    "visualization/visualize.py": """import cv2
import matplotlib.pyplot as plt

def show_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
""",
    "analysis/severity.py": """def infection_severity(infection_percent):
    if infection_percent < 20:
        return "Mild"
    elif infection_percent < 50:
        return "Moderate"
    else:
        return "Severe"
""",
    "crop_identifier.py": """# Placeholder for crop identification logic
def identify_crop(img_path):
    # Implement crop recognition (CNN or feature-based)
    return "apple"
"""
}

def create_file(path, content=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    # -------- COMMON MODULAR STRUCTURE --------
    common_dir = os.path.join(BASE_DIR, "common")
    os.makedirs(common_dir, exist_ok=True)
    create_file(os.path.join(common_dir, "__init__.py"))

    for relative_path, content in COMMON_FILES.items():
        full_path = os.path.join(common_dir, relative_path)
        create_file(full_path)
        # add __init__.py in subfolders
        subfolder = os.path.dirname(full_path)
        init_file = os.path.join(subfolder, "__init__.py")
        create_file(init_file)

    # -------- CROPS --------
    for crop, classes in CROPS_CLASSES.items():
        crop_dir = os.path.join(BASE_DIR, crop)
        os.makedirs(crop_dir, exist_ok=True)
        create_file(os.path.join(crop_dir, "__init__.py"))

        # Disease classes file
        create_file(
            os.path.join(crop_dir, f"{crop}_classes.py"),
            f"# Disease classes for {crop.upper()}\nCLASSES = {classes}\n"
        )

        # Preprocessing placeholder
        create_file(
            os.path.join(crop_dir, f"{crop}_preprocessing.py"),
            f"# Data preprocessing for {crop.upper()}\n# Reads only {crop}/train, {crop}/validation folders\n"
        )

        # Train file placeholder
        create_file(
            os.path.join(crop_dir, f"{crop}_train.py"),
            f"from common.models.model_base import build_model\n# Train {crop} disease classifier here\n"
        )

        # Predict file placeholder
        create_file(
            os.path.join(crop_dir, f"{crop}_predict.py"),
            f"# Predict disease for {crop.upper()}\n"
        )

        # Test file placeholder
        create_file(
            os.path.join(crop_dir, f"{crop}_test.py"),
            f"# Test {crop.upper()} disease model\n"
        )

        # Segmentation file placeholder
        create_file(
            os.path.join(crop_dir, f"{crop}_segmentation.py"),
            f"# Segment diseased areas in {crop.upper()} leaves\n"
        )

    # -------- MAIN --------
    create_file(
        os.path.join(BASE_DIR, "main.py"),
        "# Main pipeline\n# 1. Predict crop\n# 2. Route to crop-specific disease model\n"
    )

    print("✅ Multi-crop project structure created with modular common/ and crop-specific files!")

if __name__ == "__main__":
    main()
