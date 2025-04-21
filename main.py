import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {image_path}")
    return image

def resize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height))

def image_preprocessing(image, bin_thresh=125):
    kernel = np.ones((3, 3), np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY_INV)

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel=kernel, iterations=5)
    edges = cv2.Canny(closed, threshold1=bin_thresh, threshold2=bin_thresh + 50)

    #dilated = cv2.dilate(edges, kernel=kernel, iterations=1)

    return {
        "Original": image,
        "Graustufen": gray,
        "Binarisiert": binary,
        "Closing": closed,
        "Canny Edges": edges
    }

# --- Einstellungen ---
scale_percent = 30
input_image_path = "images/inpus.jpg"

# --- Ausführung ---
image = load_image(input_image_path)
resized_image = resize(image, scale_percent)
processed_images = image_preprocessing(resized_image)

# --- Darstellung ---
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
