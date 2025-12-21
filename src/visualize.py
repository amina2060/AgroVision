# src/visualize.py
import matplotlib.pyplot as plt
import cv2

def show_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    img_path = input("Enter image path to visualize: ")
    show_image(img_path)
