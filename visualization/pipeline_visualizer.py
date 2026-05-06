"""
Script demo pipeline 6-window cho mon CV.

Hien dong thoi 6 cua so song song:
1. Original (BGR)
2. Grayscale
3. Gaussian Blur
4. Canny Edges
5. Skeleton (MediaPipe)
6. Final output (gesture + UI)

Cach dung:
    python -m visualization.pipeline_visualizer

Dung de:
- Quay video demo cho mon CV (chung minh co day du CV pipeline)
- Screenshots cho report
- Demo truc tiep cho hoi dong cham
"""

import time
import cv2
import numpy as np

import config
from src.camera import Camera
from src.cv_operations import (
    Resizer, Grayscale, GaussianFilter, EdgeDetector, ColorConverter
)
from src.detector import PoseHandDetector
from src.feature_extractor import FeatureExtractor
from src.pca_classifier import PCAClassifier
from src.display import Display


def add_label(img, text, color=(0, 255, 0)):
    """Them label len ca tren cua tung cua so."""
    img = img.copy()
    if len(img.shape) == 2:  # gray
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def make_grid(images, rows, cols, cell_size=(320, 180)):
    """Ghep nhieu anh vao 1 grid de hien thi tat ca trong 1 cua so."""
    cw, ch = cell_size
    grid = np.zeros((ch * rows, cw * cols, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(img, (cw, ch))
        r, c = i // cols, i % cols
        grid[r*ch:(r+1)*ch, c*cw:(c+1)*cw] = resized

    return grid


def main():
    print("=" * 60)
    print("PIPELINE VISUALIZER — Demo 6-window cho CV")
    print("=" * 60)
    print("\nHien 6 cua so song song:")
    print("  1. Original")
    print("  2. Grayscale")
    print("  3. Gaussian Blur")
    print("  4. Canny Edges")
    print("  5. Skeleton (MediaPipe)")
    print("  6. Final (gesture + UI)")
    print("\nNhan ESC de thoat.\n")

    cam = Camera()
    resizer = Resizer()
    grayscale = Grayscale()
    gaussian = GaussianFilter()
    edge = EdgeDetector()
    color_conv = ColorConverter()
    detector = PoseHandDetector()
    extractor = FeatureExtractor()
    display = Display()

    # Try load classifier
    try:
        classifier = PCAClassifier.load()
        has_classifier = True
    except FileNotFoundError:
        print("[Warning] Khong co PCA model. Khong nhan dien gesture.")
        classifier = None
        has_classifier = False

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        timings = {}
        t_start = time.perf_counter()

        # Step 1: Capture (already done)
        # Step 2a: Resize
        bgr = resizer.process(frame)
        # Step 2b: Grayscale
        gray = grayscale.process(bgr)
        # Step 2c: Gaussian blur
        gray_blurred = gaussian.process(gray)
        bgr_blurred = gaussian.process(bgr)
        # Step 2d: Canny
        edges = edge.process(gray_blurred)
        # Step 2e: BGR -> RGB
        rgb = color_conv.process(bgr_blurred)
        # Step 3: Detect
        detection = detector.detect(rgb)
        # Step 4: Features
        features = extractor.extract(detection)
        # Step 5: Classify
        if features is not None and has_classifier:
            gesture, confidence = classifier.predict(features)
        else:
            gesture, confidence = 'idle', 0.0

        timings['total'] = (time.perf_counter() - t_start) * 1000

        # Skeleton overlay
        skeleton_only = bgr_blurred.copy()
        skeleton_only = detector.draw_skeleton(skeleton_only, detection)

        # Final UI
        final = bgr_blurred.copy()
        final = detector.draw_skeleton(final, detection)
        final = display.render(
            final,
            gesture=gesture,
            confidence=confidence,
            timings=timings,
        )

        # Add labels va ghep grid
        panels = [
            add_label(bgr, '1. Original (BGR)', (255, 255, 255)),
            add_label(gray, '2. Grayscale', (200, 200, 200)),
            add_label(gray_blurred, '3. Gaussian Blur', (200, 200, 255)),
            add_label(edges, '4. Canny Edges', (255, 200, 200)),
            add_label(skeleton_only, '5. Skeleton (MediaPipe)', (200, 255, 200)),
            add_label(final, '6. Final (gesture+UI)', (0, 255, 255)),
        ]

        grid = make_grid(panels, rows=2, cols=3, cell_size=(480, 270))
        cv2.imshow('Pipeline Visualizer (CV demo)', grid)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()