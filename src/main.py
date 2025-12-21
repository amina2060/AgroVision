# src/main.py
from predict import load_leaf_model, predict_leaf
from visualize import show_image
from severity import infection_severity

def main():
    model = load_leaf_model()
    while True:
        print("\n--- Leaf Disease Detection ---")
        print("1. Predict leaf disease")
        print("2. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            img_path = input("Enter image path: ")
            show_image(img_path)
            class_idx, confidence = predict_leaf(img_path, model)
            print(f"Predicted class: {class_idx}, Confidence: {confidence:.2f}")
            infection = float(input("Enter infection % for severity calculation: "))
            print("Severity level:", infection_severity(infection))
        elif choice == '2':
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()

