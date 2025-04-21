import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------- 1. Bildaufnahme -----------------
def load_and_resize(image_path, scale_percent):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Bild konnte nicht geladen werden.")
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height))

# ----------------- 2. Bildvorverarbeitung -----------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# ----------------- 3. Kantenextraktion -----------------
def extract_edges(gray, thresh):
    kernel = np.ones((3, 3), np.uint8)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
    edges = cv2.Canny(closed, 130, 180)
    return binary, closed, edges

# ----------------- Hauptablauf -----------------
input_path = "images/inpus.jpg"
scale_percent = 30
thresh = 125

image = load_and_resize(input_path, scale_percent)
gray = preprocess_image(image)
binary, closed, edges = extract_edges(gray, thresh)

# ----------------- Darstellung -----------------
processed_images = {
    "Original": image,
    "Graustufen": gray,
    "Binarisiert": binary,
    "Closing": closed,
    "Kanten (Canny)": edges
}

fig, axs = plt.subplots(3, 3, figsize=(20, 12), facecolor='white')
axs = axs.flatten()

for idx, (title, img) in enumerate(processed_images.items()):
    cmap = 'gray' if len(img.shape) == 2 else None
    axs[idx].imshow(img, cmap=cmap)
    axs[idx].set_title(f"{idx+1}. {title}", fontsize=12)
    axs[idx].axis('off')
    axs[idx].set_facecolor('white')

# Leere Felder ausblenden
for ax in axs[len(processed_images):]:
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.15)
plt.show()