# Imports
import cv2
import numpy as np
import os
from imutils import perspective

# 1. Bild laden und skalieren
def load_and_resize(image_path, scale=0.4):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild '{image_path}' konnte nicht geladen werden.")
    h, w = image.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size)

# 2a. Vorverarbeitung für A4-Blatt
def preprocess_image(image, bin_thresh=120, morph_iter=10, canny1=180, canny2=210):
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Graustufenbild
    _, binary = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY_INV) # Invertierter binärer Schwellwert
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter) # Lücken füllen
    edges = cv2.Canny(closed, canny1, canny2) # Kanten finden
    dilated = cv2.dilate(edges, kernel, iterations=1) # Kanten verbreitern
    return dilated, closed, edges

# 2b. Vorverarbeitung für Objekt
def processed_object(image, blur_kernel=(7, 7), canny1=30, canny2=80, iters=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Graustufen
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0) # Weichzeichnen
    edged = cv2.Canny(blurred, canny1, canny2) # Kanten finden
    dilated = cv2.dilate(edged, None, iterations=iters) # Kanten verstärken
    eroded = cv2.erode(dilated, None, iterations=iters) # Rauschen entfernen
    return eroded

# 3a. A4-Kontur erkennen und entzerren
def find_a4_by_contour(processed_image, original_image, epsilon=20):
    # Konturen suchen
    cnts, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise Exception("Keine Konturen gefunden!")
    largest = max(cnts, key=cv2.contourArea)
    # Näherung als Polygon
    polygon = cv2.approxPolyDP(largest, epsilon=epsilon, closed=True)
    if len(polygon) != 4:
        raise Exception("A4-Kontur hat nicht genau 4 Ecken!")
    ordered_box = perspective.order_points(polygon.reshape(4, 2)) # Eckpunkte sortieren
    warped = perspective.four_point_transform(original_image, ordered_box) # Perspektivisch entzerren

    # A4-Rand in warped_img hinzufügen - A4-Blatt rechteckig
    #cv2.rectangle(warped, (0, 0), (warped.shape[1] - 1, warped.shape[0] - 1), (0, 255, 0), 2)

    # A4-Kontur im Originalbild einzeichnen
    annotated = original_image.copy()
    cv2.drawContours(annotated, [np.int32(ordered_box)], -1, (0, 255, 0), 2)
    return warped, annotated

# 3b. A4-Kontur mit Hough Transformation erkennen
def apply_hough_lines(image, processed_image, pixels_per_mm, mode="center"):
    img = image.copy()
    if mode == "center":
        threshold, min_line_length, max_line_gap = 15, 10, 5
    elif mode == "a4":
        threshold, min_line_length, max_line_gap = 50, 100, 30
    else:
        raise ValueError("Unbekannter Modus für apply_hough_lines")
    lines = cv2.HoughLinesP(processed_image, rho=1, theta=np.pi/180,
                            threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return None, img
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    selected_lines = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = np.hypot(x2 - x1, y2 - y1)
        if mode == "center":
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            if abs(mid_x - cx) < 0.25 * w and abs(mid_y - cy) < 0.2 * h:
                selected_lines.append((length, (x1, y1, x2, y2)))
        elif mode == "a4":
            if length > 0.5 * min(w, h):
                selected_lines.append((length, (x1, y1, x2, y2)))
    if not selected_lines:
        return None, img
    if mode == "center":
        selected_lines.sort(key=lambda x: x[0], reverse=True)
        longest = selected_lines[0]
        shortest = selected_lines[-1] if len(selected_lines) > 4 else selected_lines[-1]
        for length, (x1, y1, x2, y2) in [longest, shortest]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if pixels_per_mm:
                length_mm = px_to_mm(length, pixels_per_mm)
                draw_line_with_text(img, x1, y1, x2, y2, length_mm)
    elif mode == "a4":
        for length, (x1, y1, x2, y2) in selected_lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return selected_lines, img

# 4. Objekte im entzerrten Bild erkennen und messen
def detect_objects_contour(image, pixels_per_mm, border_thresh=10):
    processed = processed_object(image)
    cnts = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    h, w = image.shape[:2]
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        # Objekte an Rand überspringen
        if x <= border_thresh or y <= border_thresh or (x + bw) >= (w - border_thresh) or (y + bh) >= (h - border_thresh):
            continue
        # Pixel in mm umrechnen
        width_mm = px_to_mm(bw, pixels_per_mm)
        height_mm = px_to_mm(bh, pixels_per_mm)
        # Rechteck & Maße einzeichnen
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        draw_line_with_text(image, x, y + bh, x + bw, y + bh, width_mm) # Breite unten
        draw_line_with_text(image, x + bw, y, x + bw, y + bh, height_mm) # Höhe rechts
    return image

# 5. Umrechnungsfaktor Pixel in mm anhand A4-Seite
def calculate_pixels_per_mm(image_shape, a4_width_mm=210, a4_height_mm=297):
    h, w = image_shape[:2]
    return (w / a4_width_mm + h / a4_height_mm) / 2

def px_to_mm(pixel_length, pixels_per_mm):
    return pixel_length / pixels_per_mm

# Textbeschriftung der Länge direkt ins Bild schreiben
def draw_line_with_text(img, x1, y1, x2, y2, length_mm):
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    cv2.putText(img, f"{length_mm:.1f}mm", (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

# Speicherordner anlegen, falls nicht vorhanden
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Zwischenschritte speichern
def save_steps(steps, output_dir="output_steps"):
    ensure_dir(output_dir)
    for name, img in steps.items():
        path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(path, img)
        print(f"Gespeichert: {path}")

# Hauptfunktion: Bild laden, A4 finden, entzerren, messen
def run(image_path):
    img = load_and_resize(image_path)
    preprocessed, closed, edges = preprocess_image(img)

    #pixels_per_mm = calculate_pixels_per_mm(img.shape)
    #a4_lines, a4_img = apply_hough_lines(img, dilated, pixels_per_mm=pixels_per_mm, mode="a4")
    #center_lines, hough_img = apply_hough_lines(img, dilated, pixels_per_mm=pixels_per_mm, mode="center")

    warped_img, a4_annotated = find_a4_by_contour(preprocessed, img)
    pixels_per_mm = calculate_pixels_per_mm(warped_img.shape)
    detected_img = detect_objects_contour(warped_img.copy(), pixels_per_mm)

    steps = {
        "Original_resized": img,
        "Closing": closed,
        "Canny": edges,
        "A4_Contour_detected": a4_annotated,
        "Warped_A4": warped_img,
        "Detected_Objects_with_mm": detected_img
    }
    save_steps(steps)

if __name__ == '__main__':
    run("images/inbus2.jpg")