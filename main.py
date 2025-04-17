import cv2
import matplotlib.pyplot as plt

# --- Schritt 1: Bild laden ---
image = cv2.imread("images/inpus-1.jpg")
orig = image.copy()

# --- Schritt 2: Bild vorverarbeiten ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)


# Anzeige der Bilder
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
axs[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[0].axis('off')

axs[1].imshow(gray, cmap='gray')
axs[1].set_title('Graustufen')
axs[1].axis('off')

plt.tight_layout()
plt.show()

