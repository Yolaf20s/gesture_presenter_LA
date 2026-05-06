"""
Class Camera: doc video tu webcam hoac DroidCam.
Step 1 cua pipeline.
"""

import cv2
import config


class Camera:
    """Wrapper quanh cv2.VideoCapture, doc config va xu ly loi."""

    def __init__(self, source=None, name='main'):
        """
        source: camera index (so) hoac URL (IP camera).
                Neu None, lay tu config.CAMERA_SOURCE.
        name:   ten de log (huu ich khi co nhieu camera).
        """
        self.source = source if source is not None else config.CAMERA_SOURCE
        self.name = name
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[Camera-{name}] Khong mo duoc camera source={self.source}. "
                f"Kiem tra: DroidCam co dang chay khong? Index dung khong?"
            )

        # Set resolution & FPS theo config
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        # Doc info thuc te (camera co the khong support resolution duoc set)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(
            f"[Camera-{name}] Da mo source={self.source} "
            f"-> {self.width}x{self.height} @ {self.fps:.0f}fps"
        )

    def read_frame(self):
        """
        Doc 1 frame.
        Return: numpy array shape (H, W, 3) hoac None neu loi.
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        """Giai phong camera."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print(f"[Camera-{self.name}] Da release.")

    def __del__(self):
        """Tu dong release khi object bi destroy."""
        try:
            self.release()
        except Exception:
            pass


# ============================================================
# Test khi chay file truc tiep
# ============================================================
if __name__ == '__main__':
    """Chay file nay de test class Camera nhanh."""
    cam = Camera()
    print("Nhan ESC de thoat.")
    while True:
        frame = cam.read_frame()
        if frame is None:
            print("Loi doc frame!")
            break
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    cam.release()
    cv2.destroyAllWindows()