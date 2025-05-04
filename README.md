# Automatische Objektvermessung mit OpenCV

Dieses Projekt hat zum Ziel, metallische Werkstücke (z.B. Inbusschlüssel) automatisiert anhand eines Kamerabilds auszumessen.

## Installation & Setup

1. **Virtuelle Umgebung erstellen**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Benötigte Bibliotheken installieren**
   ```bash
   pip install opencv-python matplotlib numpy
   ```

## Problemstellung

In modernen Fräsmaschinen kann ein falsch dimensionierter Rohling zur Kollision mit der Spindel führen – dies verursacht teure Schäden. Ziel ist es daher, eine automatische „**Smarte Aufspannkontrolle**“ zu entwickeln, die:

- Werkstücke auf Bildern erkennt,
- deren Kantenlängen misst,
- und das Ergebnis in **Millimeter** ausgibt.

## Vorgehensweise
![grafik](https://github.com/user-attachments/assets/4c45e473-2206-4093-8e09-4b455a624856)

Zur präzisen Vermessung wird folgende Bildverarbeitungskette eingesetzt:

1. **A4-Referenz erkennen**
   Ein A4-Blatt dient als metrische Referenz. Es wird über Kantenerkennung und Konturenlokalisierung im Bild erkannt.

2. **Perspektivische Entzerrung (Top-Down-Ansicht)**
   Die A4-Kontur wird genutzt, um das Bild perspektivisch zu entzerren. Dadurch kann das A4-Blatt als maßstabsgetreues Rechteck angenommen werden.

3. **Objekterkennung per Konturanalyse**
   Im entzerrten Bild werden relevante Objekte erkannt, Randbereiche ignoriert, und mit **Bounding Boxes** versehen.

4. **Umrechnung von Pixel in Millimeter**
   Da die reale Größe eines A4-Blattes bekannt ist (210 × 297 mm), wird die Auflösung im Bild berechnet:

   ```python
   px_per_mm_w = warped_width / 210
   px_per_mm_h = warped_height / 297
   pixels_per_mm = (px_per_mm_w + px_per_mm_h) / 2
   ```

   Mit dieser Auflösung werden erkannte Objektkanten umgerechnet:

   ```python
   mm = pixel_length / pixels_per_mm
   ```

   Die gemessenen Werte werden direkt im Bild als **Längenbeschriftung** eingeblendet.

## Besonderheiten

- Die größte Herausforderung besteht in der präzisen Vermessung von asymmetrischen Objekten wie einem Inbusschlüssel.
- Lichtverhältnisse und Bildschärfe haben entscheidenden Einfluss auf die Genauigkeit der Ergebnisse.
- Erweiterbar mit Kalibrierung über bekannte Objekte (z.B. Münze, Papierkante).

## Referenzen
- [Contour Detektion](https://learnopencv.com/contour-detection-using-opencv-python-c/)
- [Measuring Size - Bounding Box](https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/)
- [Contour Approximation](https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/)
- [Grafik zeichnen](https://excalidraw.com/)
