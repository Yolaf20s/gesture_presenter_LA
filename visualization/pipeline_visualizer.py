"""
Visual pipeline demo for presentations and reports.

The final gesture panel now follows the same inference path as the real runtime:
- resize -> RGB -> MediaPipe -> features -> PCA classifier -> trigger logic

The grayscale / blur / edge panels remain educational side views only.
"""

import time

import cv2
import numpy as np

from src.camera import Camera
from src.cv_operations import EdgeDetector, GaussianFilter, Grayscale, Resizer
from src.detector import PoseHandDetector
from src.display import Display
from src.feature_extractor import FeatureExtractor
from src.gesture_stabilizer import GestureStabilizer
from src.pca_classifier import PCAClassifier
from src.preprocessor import Preprocessor


def add_label(img, text, color=(0, 255, 0)):
    img = img.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def make_grid(images, rows, cols, cell_size=(320, 180)):
    cw, ch = cell_size
    grid = np.zeros((ch * rows, cw * cols, 3), dtype=np.uint8)

    for index, img in enumerate(images):
        if index >= rows * cols:
            break
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(img, (cw, ch))
        row, col = index // cols, index % cols
        grid[row * ch:(row + 1) * ch, col * cw:(col + 1) * cw] = resized

    return grid


def main():
    print("=" * 60)
    print("PIPELINE VISUALIZER")
    print("=" * 60)
    print("\nPanels:")
    print("  1. Original")
    print("  2. Grayscale")
    print("  3. Gaussian Blur")
    print("  4. Canny Edges")
    print("  5. Skeleton")
    print("  6. Final runtime output")
    print("\nPress ESC to quit.\n")

    cam = Camera()
    pre = Preprocessor()
    resizer = Resizer()
    grayscale = Grayscale()
    gaussian = GaussianFilter()
    edge = EdgeDetector()
    detector = PoseHandDetector()
    extractor = FeatureExtractor()
    stabilizer = GestureStabilizer()
    display = Display()

    try:
        classifier = PCAClassifier.load()
        has_classifier = True
    except FileNotFoundError:
        print("[Warning] No PCA model found. Gesture labels will stay idle.")
        classifier = None
        has_classifier = False

    fps_window = []
    last_t = time.time()

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        timings = {}
        t_start = time.perf_counter()

        # Runtime path
        rgb_runtime, bgr_runtime = pre.process(frame)
        detection = detector.detect(rgb_runtime)
        features, feature_meta = extractor.extract(detection, return_meta=True)
        tracking_state = feature_meta["tracking_state"]
        raw_gesture = "idle"
        raw_confidence = 0.0
        if features is not None and has_classifier:
            raw_gesture, raw_confidence = classifier.predict(features)
        gesture, confidence = stabilizer.update(raw_gesture, raw_confidence, tracking_state)

        # Educational side panels
        resized = resizer.process(frame)
        gray = grayscale.process(resized)
        gray_blurred = gaussian.process(gray)
        edges = edge.process(gray_blurred)
        timings["total"] = (time.perf_counter() - t_start) * 1000
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps_window.append(1.0 / dt)
            if len(fps_window) > 30:
                fps_window.pop(0)
        fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

        skeleton_only = detector.draw_skeleton(bgr_runtime.copy(), detection)
        final = detector.draw_skeleton(bgr_runtime.copy(), detection)
        final = display.render(
            final,
            gesture=gesture,
            confidence=confidence,
            timings=timings,
            tracking_state=tracking_state,
            fps=fps,
        )

        panels = [
            add_label(frame, "1. Original", (255, 255, 255)),
            add_label(gray, "2. Grayscale", (200, 200, 200)),
            add_label(gray_blurred, "3. Gaussian Blur", (200, 200, 255)),
            add_label(edges, "4. Canny Edges", (255, 200, 200)),
            add_label(skeleton_only, "5. Skeleton", (200, 255, 200)),
            add_label(final, "6. Final runtime output", (0, 255, 255)),
        ]

        grid = make_grid(panels, rows=2, cols=3, cell_size=(480, 270))
        cv2.imshow("Pipeline Visualizer", grid)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
