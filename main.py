import cv2
import numpy as np
import os

# ----------------- 1. Bildaufnahme & Resize -----------------
def load_and_resize(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild '{image_path}' konnte nicht geladen werden.")
    h, w = image.shape[:2]
    new_w = int(w * 0.4)
    new_h = int(h * 0.4)
    return cv2.resize(image, (new_w, new_h))

# ----------------- 2. Bildvorverarbeitung -----------------
def preprocess_image(image, bin_thresh):
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=10)
    edges = cv2.Canny(closed, bin_thresh + 50, bin_thresh + 130)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return binary, closed, edges, dilated

# ----------------- 3a. Hough-Transformation -----------------
def apply_hough_lines(image, processed_image, pixels_per_mm, mode="center"):
    img = image.copy()
    if mode == "center":
        threshold, min_line_length, max_line_gap = 15, 10, 5
    elif mode == "a4":
        threshold, min_line_length, max_line_gap = 50, 100, 20
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
            if abs(mid_x - cx) < 0.25 * w and abs(mid_y - cy) < 0.25 * h:
                selected_lines.append((length, (x1, y1, x2, y2)))
        elif mode == "a4":
            if length > 0.5 * min(w, h):
                selected_lines.append((length, (x1, y1, x2, y2)))

    if not selected_lines:
        return None, img

    if mode == "center":
        selected_lines.sort(key=lambda x: x[0], reverse=True)
        longest = selected_lines[0]
        shortest = selected_lines[4] if len(selected_lines) > 4 else selected_lines[-1]
        for length, (x1, y1, x2, y2) in [longest, shortest]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if pixels_per_mm:
                length_mm = px_to_mm(length, pixels_per_mm)
                draw_line_with_text(img, x1, y1, x2, y2, length_mm)
    elif mode == "a4":
        for length, (x1, y1, x2, y2) in selected_lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return selected_lines, img

# ----------------- 4. Längenmessung in mm -----------------
def calculate_pixels_per_mm(image_shape, a4_width_mm=210, a4_height_mm=297):
    h, w = image_shape[:2]
    avg_pixels_per_mm_width = w / a4_width_mm
    avg_pixels_per_mm_height = h / a4_height_mm
    return (avg_pixels_per_mm_width + avg_pixels_per_mm_height) / 2

def px_to_mm(pixel_length, pixels_per_mm):
    return pixel_length / pixels_per_mm

def draw_line_with_text(img, x1, y1, x2, y2, length_mm):
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    cv2.putText(img,
                f"{length_mm:.1f}mm",
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA)

# ----------------- Speichern -----------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_steps(steps, output_dir="output_steps"):
    ensure_dir(output_dir)
    for name, img in steps.items():
        # für Graustufenbilder in BGR zurückwandeln
        if img.ndim == 2:
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            out = img
        path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(path, out)
        print(f"Gespeichert: {path}")

# ----------------- Hauptablauf -----------------
def main():
    img = load_and_resize(image_path="images/inpus-4.jpg")
    binary, closed, edges, dilated = preprocess_image(img, bin_thresh=100)

    pixels_per_mm = calculate_pixels_per_mm(img.shape)
    a4_lines, a4_img = apply_hough_lines(img, dilated, pixels_per_mm=pixels_per_mm, mode="a4")
    center_lines, hough_img = apply_hough_lines(img, dilated, pixels_per_mm=pixels_per_mm, mode="center")

    steps = {
        "Original_resized": img,
        "Binary": binary,
        "Closing": closed,
        "Canny": edges,
        "Dilated": dilated,
        "A4_detected_Hough": a4_img,
        "Hough_Lines_Center": hough_img
    }
    save_steps(steps)

if __name__ == '__main__':
    main()