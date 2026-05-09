"""
Main runtime pipeline.

Shared by:
- the simple OpenCV CLI preview
- the Tkinter GUI
"""

import time

import cv2
import numpy as np

import config
from src.action_trigger import ActionTrigger
from src.camera import Camera
from src.detector import PoseHandDetector
from src.display import Display
from src.feature_extractor import FeatureExtractor
from src.gesture_stabilizer import GestureStabilizer
from src.pca_classifier import PCAClassifier
from src.preprocessor import Preprocessor


class Pipeline:
    """Read a frame, classify the gesture, and optionally trigger an action."""

    def __init__(self, camera_source=None):
        self.camera = Camera(source=camera_source)
        self.preprocessor = Preprocessor()
        self.detector = PoseHandDetector()
        self.feature_extractor = FeatureExtractor()
        self.stabilizer = GestureStabilizer()
        self.action_trigger = ActionTrigger()
        self.display = Display()

        try:
            self.classifier = PCAClassifier.load()
            self.has_classifier = True
        except FileNotFoundError:
            self.classifier = None
            self.has_classifier = False
            print(
                "[Pipeline] PCA model not found. Running in preview-only mode "
                "(no gesture classification)."
            )

        self._fps_window = []
        self._last_step_t = time.time()

        print("[Pipeline] Ready.")

    def step(self, show_overlay=False):
        """Process one frame and return a result dictionary."""
        timings = {}
        t_total_start = time.time()

        t = time.time()
        frame = self.camera.read_frame()
        timings["capture"] = (time.time() - t) * 1000
        if frame is None:
            return None

        t = time.time()
        rgb, bgr = self.preprocessor.process(frame)
        timings["preprocess"] = (time.time() - t) * 1000

        t = time.time()
        detection = self.detector.detect(rgb)
        timings["detect"] = (time.time() - t) * 1000

        t = time.time()
        features, feature_meta = self.feature_extractor.extract(
            detection,
            return_meta=True,
        )
        tracking_state = feature_meta["tracking_state"]
        timings["features"] = (time.time() - t) * 1000

        t = time.time()
        features_proj = None
        raw_gesture = "idle"
        raw_confidence = 0.0
        if features is not None and self.has_classifier:
            raw_gesture, raw_confidence = self.classifier.predict(features)
            features_proj = self.classifier.project(features)
        stable_gesture, stable_confidence = self.stabilizer.update(
            raw_gesture,
            raw_confidence,
            tracking_state,
        )
        timings["classify"] = (time.time() - t) * 1000

        t = time.time()
        triggered, message = self.action_trigger.trigger(
            stable_gesture,
            stable_confidence,
            tracking_state=tracking_state,
        )
        timings["trigger"] = (time.time() - t) * 1000

        t = time.time()
        annotated = self.detector.draw_skeleton(bgr.copy(), detection)
        cooldown_remaining = self.action_trigger.get_cooldown_remaining()
        timings["display"] = (time.time() - t) * 1000

        timings["total"] = (time.time() - t_total_start) * 1000
        fps = self._update_fps()

        if show_overlay:
            annotated = self.display.render(
                annotated,
                gesture=stable_gesture,
                confidence=stable_confidence,
                timings=timings,
                tracking_state=tracking_state,
                fps=fps,
                cooldown_remaining=cooldown_remaining,
                just_triggered=triggered,
                trigger_message=message,
            )

        return {
            "frame": annotated,
            "features": features,
            "features_proj": features_proj,
            "gesture": stable_gesture,
            "confidence": stable_confidence,
            "raw_gesture": raw_gesture,
            "raw_confidence": raw_confidence,
            "tracking_state": tracking_state,
            "feature_meta": feature_meta,
            "timings": timings,
            "fps": fps,
            "cooldown_remaining": cooldown_remaining,
            "just_triggered": triggered,
            "trigger_message": message if triggered else "",
            "trigger_status": message,
        }

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_step_t
        self._last_step_t = now
        if dt > 0:
            self._fps_window.append(1.0 / dt)
            if len(self._fps_window) > 30:
                self._fps_window.pop(0)
        return float(np.mean(self._fps_window)) if self._fps_window else 0.0

    def run(self):
        """CLI mode with `cv2.imshow`. Press ESC to quit."""
        print(f"\n{'=' * 60}")
        print("Gesture Presenter is running. Press ESC to quit.")
        print(f"{'=' * 60}\n")

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, 1280, 720)

        last_log = time.time()
        try:
            while True:
                result = self.step(show_overlay=True)
                if result is None:
                    print("[Pipeline] Failed to read a frame. Exiting.")
                    break

                cv2.imshow(config.WINDOW_NAME, result["frame"])

                if time.time() - last_log > 5:
                    print(
                        f"[Pipeline] FPS: {result['fps']:.1f} | "
                        f"total: {result['timings']['total']:.1f}ms | "
                        f"gesture: {result['gesture']} "
                        f"({result['confidence'] * 100:.0f}%) | "
                        f"tracking: {result['tracking_state']}"
                    )
                    last_log = time.time()

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n[Pipeline] Cleaning up...")
        try:
            self.camera.release()
        except Exception:
            pass
        try:
            self.detector.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Pipeline] Done.")


if __name__ == "__main__":
    Pipeline().run()
