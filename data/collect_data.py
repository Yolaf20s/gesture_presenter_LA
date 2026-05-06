"""
Script thu thap dataset cho PCA training.

Cach dung:
    python -m data.collect_data

Workflow:
1. Script lan luot doi tung gesture: right_arm, left_arm, ...
2. Voi moi gesture:
   - Hien preview camera
   - Bam SPACE de bat dau thu
   - 3 giay countdown
   - Thu 100 samples (auto, ~5 giay)
   - Bam 's' de skip gesture nay (neu khong muon thu)
3. Save tat ca features ra file .npy
"""

import os
import time
import cv2
import numpy as np

import config
from src.camera import Camera
from src.preprocessor import Preprocessor
from src.detector import PoseHandDetector
from src.feature_extractor import FeatureExtractor


def collect_dataset(person_name='user1'):
    """Thu dataset cho 1 nguoi."""
    cam = Camera()
    pre = Preprocessor()
    detector = PoseHandDetector()
    extractor = FeatureExtractor()

    # Tao folder de luu
    save_dir = os.path.join(config.DATASET_DIR, person_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"COLLECT DATASET cho: {person_name}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    for gesture in config.GESTURE_CLASSES:
        save_path = os.path.join(save_dir, f"{gesture}.npy")

        # Skip neu da thu roi
        if os.path.exists(save_path):
            choice = input(
                f"[{gesture}] Da co file {save_path}. "
                f"Thu lai? (y/N): "
            ).strip().lower()
            if choice != 'y':
                print(f"  Skip {gesture}.")
                continue

        print(f"\n--- Gesture: {gesture} ---")
        print(f"Hay lam dong tac '{gesture}'.")
        print(f"  - SPACE: bat dau thu {config.SAMPLES_PER_GESTURE} samples")
        print(f"  - s: skip gesture nay")
        print(f"  - ESC: thoat hoan toan\n")

        # Phase 1: preview, doi user san sang
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
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2,
            )
            cv2.putText(
                annotated,
                "SPACE: start | s: skip | ESC: quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1,
            )

            cv2.imshow('Collecting', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                break
            elif key == ord('s'):
                print(f"  Skipped {gesture}.")
                gesture = None
                break
            elif key == 27:  # ESC
                print("\nUser exit.")
                cam.release()
                detector.close()
                cv2.destroyAllWindows()
                return

        if gesture is None:
            continue

        # Phase 2: countdown 3-2-1
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
                    annotated, f"START IN {n}...",
                    (annotated.shape[1] // 2 - 100, annotated.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
                )
                cv2.imshow('Collecting', annotated)
                cv2.waitKey(1)

        # Phase 3: thu samples
        features_list = []
        target = config.SAMPLES_PER_GESTURE
        print(f"  Dang thu {target} samples...")

        while len(features_list) < target:
            frame = cam.read_frame()
            if frame is None:
                continue
            rgb, bgr = pre.process(frame)
            detection = detector.detect(rgb)
            features = extractor.extract(detection)

            if features is not None:
                features_list.append(features)

            annotated = detector.draw_skeleton(bgr.copy(), detection)
            progress = len(features_list)
            cv2.putText(
                annotated,
                f"COLLECTING: {progress}/{target}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2,
            )

            # Progress bar
            bar_w = annotated.shape[1] - 20
            filled = int(bar_w * progress / target)
            cv2.rectangle(annotated, (10, 50), (10 + bar_w, 65),
                          (50, 50, 50), -1)
            cv2.rectangle(annotated, (10, 50), (10 + filled, 65),
                          (0, 255, 0), -1)

            cv2.imshow('Collecting', annotated)
            cv2.waitKey(1)

        # Save
        features_array = np.array(features_list, dtype=np.float32)
        np.save(save_path, features_array)
        print(f"  Saved {len(features_list)} samples -> {save_path}")
        print(f"  Shape: {features_array.shape}")

    cam.release()
    detector.close()
    cv2.destroyAllWindows()
    print(f"\nHoan tat thu dataset cho {person_name}!")


if __name__ == '__main__':
    name = input("Nhap ten nguoi (vd: 'an', 'binh', 'cuong'): ").strip()
    if not name:
        name = 'user1'
    collect_dataset(name)