"""
Frame preprocessing.

The runtime intentionally uses two views of the same frame:
- a resized RGB frame for MediaPipe inference
- the original-resolution BGR frame for display
"""

import cv2

import config


class Preprocessor:
    """Prepare frames for detection while keeping the display sharp."""

    def __init__(self, target_size=None):
        if target_size is None:
            target_size = (config.RESIZE_WIDTH, config.RESIZE_HEIGHT)
        self.target_size = target_size
        print(
            f"[Preprocessor] Detect at {target_size}, display at original resolution."
        )

    def process(self, frame):
        """
        Return `(rgb_small, bgr_full)`.

        `rgb_small` is resized for faster MediaPipe inference.
        `bgr_full` keeps the original resolution for display.
        """
        small_bgr = cv2.resize(frame, self.target_size)
        rgb_small = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        bgr_full = frame.copy()
        return rgb_small, bgr_full


if __name__ == "__main__":
    from src.camera import Camera

    cam = Camera()
    pre = Preprocessor()

    print("Press ESC to quit.")
    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        rgb_small, bgr_full = pre.process(frame)
        cv2.imshow("Display (original)", bgr_full)
        cv2.imshow(
            "Detection (resized)",
            cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR),
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
