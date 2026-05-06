"""
Pipeline: orchestrate 7-step CV pipeline.

Co 2 mode dung chung 1 engine:
- step(show_overlay=False): xu ly 1 frame, return dict. GUI mode (frame sach).
- run(): CLI loop voi cv2.imshow + overlay.
"""

import time
import cv2
import numpy as np

import config
from src.camera import Camera
from src.preprocessor import Preprocessor
from src.detector import PoseHandDetector
from src.feature_extractor import FeatureExtractor
from src.pca_classifier import PCAClassifier
from src.action_trigger import ActionTrigger
from src.display import Display


class Pipeline:
    """Engine: read frame -> 6 step -> annotated frame + result dict."""

    def __init__(self):
        self.camera = Camera()
        self.preprocessor = Preprocessor()
        self.detector = PoseHandDetector()
        self.feature_extractor = FeatureExtractor()
        self.action_trigger = ActionTrigger()
        self.display = Display()

        try:
            self.classifier = PCAClassifier.load()
            self.has_classifier = True
        except FileNotFoundError:
            self.classifier = None
            self.has_classifier = False
            print("[Pipeline] Khong tim thay PCA model "
                  "-> chay 'preview only' (khong classify gesture).")

        self._fps_window = []
        self._last_step_t = time.time()

        print("[Pipeline] San sang.")

    def step(self, show_overlay=False):
        """
        Process 1 frame.

        show_overlay=True  -> draw cv2 text overlay (for CLI)
        show_overlay=False -> only skeleton on frame (for GUI; clean look)
        """
        timings = {}
        t_total_start = time.time()

        t = time.time()
        frame = self.camera.read_frame()
        timings['capture'] = (time.time() - t) * 1000
        if frame is None:
            return None

        t = time.time()
        rgb, bgr = self.preprocessor.process(frame)
        timings['preprocess'] = (time.time() - t) * 1000

        t = time.time()
        detection = self.detector.detect(rgb)
        timings['detect'] = (time.time() - t) * 1000

        t = time.time()
        features = self.feature_extractor.extract(detection)
        timings['features'] = (time.time() - t) * 1000

        t = time.time()
        features_proj = None
        if features is not None and self.has_classifier:
            gesture, confidence = self.classifier.predict(features)
            x_centered = features.astype(np.float64) - self.classifier.mean
            features_proj = x_centered @ self.classifier.eigenvectors
        else:
            gesture, confidence = 'idle', 0.0
        timings['classify'] = (time.time() - t) * 1000

        t = time.time()
        triggered, message = self.action_trigger.trigger(gesture, confidence)
        timings['trigger'] = (time.time() - t) * 1000

        # Skeleton always drawn. Overlay text only for CLI mode.
        t = time.time()
        annotated = self.detector.draw_skeleton(bgr.copy(), detection)
        cooldown_remaining = self.action_trigger.get_cooldown_remaining()

        if show_overlay:
            annotated = self.display.render(
                annotated, gesture, confidence, timings,
                cooldown_remaining=cooldown_remaining,
                just_triggered=triggered,
                trigger_message=message,
            )
        timings['display'] = (time.time() - t) * 1000

        timings['total'] = (time.time() - t_total_start) * 1000

        now = time.time()
        dt = now - self._last_step_t
        self._last_step_t = now
        if dt > 0:
            self._fps_window.append(1.0 / dt)
            if len(self._fps_window) > 30:
                self._fps_window.pop(0)
        fps = float(np.mean(self._fps_window)) if self._fps_window else 0.0

        return {
            'frame': annotated,
            'features': features,
            'features_proj': features_proj,
            'gesture': gesture,
            'confidence': confidence,
            'timings': timings,
            'fps': fps,
            'cooldown_remaining': cooldown_remaining,
            'just_triggered': triggered,
            'trigger_message': message if triggered else '',
        }

    def run(self):
        """CLI mode: cv2 window + overlay. ESC to quit."""
        print(f"\n{'=' * 60}")
        print(f"Gesture Presenter dang chay. Nhan ESC de thoat.")
        print(f"{'=' * 60}\n")

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, 1280, 720)

        last_log = time.time()
        try:
            while True:
                result = self.step(show_overlay=True)  # CLI: overlay ON
                if result is None:
                    print("[Pipeline] Khong doc duoc frame, thoat.")
                    break

                cv2.imshow(config.WINDOW_NAME, result['frame'])

                if time.time() - last_log > 5:
                    print(f"[Pipeline] FPS: {result['fps']:.1f} | "
                          f"avg total: {result['timings']['total']:.1f}ms | "
                          f"gesture: {result['gesture']} "
                          f"({result['confidence'] * 100:.0f}%)")
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


if __name__ == '__main__':
    Pipeline().run()