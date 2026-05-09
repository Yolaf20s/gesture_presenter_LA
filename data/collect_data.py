"""
Interactive dataset collection for gesture training.

Usage:
    python3 -m data.collect_data
"""

from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

import config
from src.camera import Camera
from src.detector import PoseHandDetector
from src.feature_extractor import FeatureExtractor
from src.preprocessor import Preprocessor


def collect_dataset(person_name="user1"):
    """Collect one session per gesture for a single person."""
    cam = Camera()
    pre = Preprocessor()
    detector = PoseHandDetector()
    extractor = FeatureExtractor()

    save_dir = Path(config.DATASET_DIR) / person_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"COLLECT DATASET FOR: {person_name}")
    print(f"Save directory: {save_dir}")
    print(f"{'=' * 60}\n")

    try:
        for gesture in config.GESTURE_CLASSES:
            gesture_dir = save_dir / gesture
            gesture_dir.mkdir(parents=True, exist_ok=True)
            existing_sessions = sorted(gesture_dir.glob("*.npy"))
            if existing_sessions:
                choice = input(
                    f"[{gesture}] Found {len(existing_sessions)} existing session(s). "
                    "Record another one? (Y/n): "
                ).strip().lower()
                if choice == "n":
                    print(f"  Skipped {gesture}.")
                    continue

            print(f"\n--- Gesture: {gesture} ---")
            print(f"Perform the gesture '{gesture}'.")
            print(f"  - SPACE: start collecting {config.SAMPLES_PER_GESTURE} valid samples")
            print("  - s: skip this gesture")
            print("  - ESC: quit entirely\n")

            while True:
                frame = cam.read_frame()
                if frame is None:
                    continue

                rgb, bgr = pre.process(frame)
                detection = detector.detect(rgb)
                annotated = detector.draw_skeleton(bgr.copy(), detection)

                cv2.putText(
                    annotated,
                    f"Gesture: {gesture}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    "SPACE: start | s: skip | ESC: quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("Collecting", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    break
                if key == ord("s"):
                    print(f"  Skipped {gesture}.")
                    gesture = None
                    break
                if key == 27:
                    print("\nUser requested exit.")
                    return

            if gesture is None:
                continue

            for n in [3, 2, 1]:
                countdown_start = time.time()
                while time.time() - countdown_start < 1.0:
                    frame = cam.read_frame()
                    if frame is None:
                        continue
                    rgb, bgr = pre.process(frame)
                    detection = detector.detect(rgb)
                    annotated = detector.draw_skeleton(bgr.copy(), detection)

                    cv2.putText(
                        annotated,
                        f"START IN {n}...",
                        (annotated.shape[1] // 2 - 100, annotated.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3,
                    )
                    cv2.imshow("Collecting", annotated)
                    cv2.waitKey(1)

            features_list = []
            target = config.SAMPLES_PER_GESTURE
            skip_current_gesture = False
            print(f"  Collecting {target} valid samples...")

            while len(features_list) < target:
                frame = cam.read_frame()
                if frame is None:
                    continue

                rgb, bgr = pre.process(frame)
                detection = detector.detect(rgb)
                features, meta = extractor.extract(detection, return_meta=True)

                is_valid, reason = _sample_status(gesture, features, meta)
                if is_valid:
                    features_list.append(features)

                annotated = detector.draw_skeleton(bgr.copy(), detection)
                progress = len(features_list)
                status = (
                    f"{reason} | visibility={meta['min_pose_visibility']:.2f} "
                    f"| hands={meta['hands_detected']}"
                )
                cv2.putText(
                    annotated,
                    f"COLLECTING: {progress}/{target}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    status,
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                bar_w = annotated.shape[1] - 20
                filled = int(bar_w * progress / target)
                cv2.rectangle(annotated, (10, 72), (10 + bar_w, 87), (50, 50, 50), -1)
                cv2.rectangle(annotated, (10, 72), (10 + filled, 87), (0, 255, 0), -1)
                cv2.imshow("Collecting", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    print(f"  Skipped {gesture}.")
                    skip_current_gesture = True
                    break
                if key == 27:
                    print("\nUser requested exit.")
                    return

            if skip_current_gesture:
                continue

            session_name = config.SESSION_FILENAME_TEMPLATE.format(
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            save_path = gesture_dir / session_name
            features_array = np.array(features_list, dtype=np.float32)
            np.save(save_path, features_array)
            print(f"  Saved {len(features_list)} samples -> {save_path}")
            print(f"  Shape: {features_array.shape}")
    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        print(f"\nFinished dataset collection for {person_name}.")


def _sample_status(gesture, features, meta):
    if features is None or meta["tracking_state"] != "ready":
        return False, meta["tracking_state"]
    if gesture in config.HAND_REQUIRED_GESTURES and meta["hands_detected"] < 1:
        return False, "need hand"
    return True, "accepted"


if __name__ == "__main__":
    name = input("Enter a person name (for example: alex): ").strip()
    if not name:
        name = "user1"
    collect_dataset(name)
