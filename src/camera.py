"""
Camera wrapper for reading frames from a webcam, DroidCam, or IP stream.
"""

import cv2

import config


class Camera:
    """Thin wrapper around `cv2.VideoCapture` with friendly errors."""

    def __init__(self, source=None, name="main"):
        self.source = source if source is not None else config.CAMERA_SOURCE
        self.name = name
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[Camera-{name}] Could not open source={self.source}. "
                "Check that the camera is connected and the source index/URL is correct."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(
            f"[Camera-{name}] Opened source={self.source} "
            f"-> {self.width}x{self.height} @ {self.fps:.0f}fps"
        )

    def read_frame(self):
        """Read one frame. Return `None` if the backend fails."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        """Release camera resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print(f"[Camera-{self.name}] Released.")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


if __name__ == "__main__":
    cam = Camera()
    print("Press ESC to quit.")
    while True:
        frame = cam.read_frame()
        if frame is None:
            print("Failed to read a frame.")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
