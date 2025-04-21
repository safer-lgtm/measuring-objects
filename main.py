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
    edges = cv2.Canny(closed, thresh, thresh + 50)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return binary, closed, edges, dilated

# ----------------- 4. Hough-Transformation -----------------
def apply_hough_inbus(edges, image):
    img = image.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=50, minLineLength=30, maxLineGap=10)
    h, w = img.shape[:2]
    center_x = w // 2
    center_y = h // 2
    selected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            euc_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            mx = (x1 + x2) // 2
            my = (y1 + y2) // 2
            # Nur Linien im Zentrum
            if abs(mx - center_x) < 0.25 * w and abs(my - center_y) < 0.25 * h:
                selected_lines.append((euc_dist, (x1, y1, x2, y2)))
        # Nach Länge sortieren, größte zuerst
        selected_lines = sorted(selected_lines, key=lambda x: x[0], reverse=True)
        if len(selected_lines) >= 1:
            x1, y1, x2, y2 = selected_lines[0][1]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # lang = grün

        if len(selected_lines) >= 2:
            x1, y1, x2, y2 = selected_lines[-3][1]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # kurz = rot
    return img

# ----------------- Hauptablauf -----------------
input_path = "images/inpus.jpg"
scale_percent = 35
thresh = 110

image = load_and_resize(input_path, scale_percent)
gray = preprocess_image(image)
binary, closed, edges, dilated = extract_edges(gray, thresh)
hough_result =  apply_hough_inbus(dilated, image)

# ----------------- Darstellung -----------------
processed_images = {
    "Original": image,
    "Graustufen": gray,
    "Binarisiert": binary,
    "Closing": closed,
    "Kanten (Canny)": edges,
    "Hough-Linien": hough_result
}

fig, axs = plt.subplots(3, 3, figsize=(30, 20), facecolor='white')
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