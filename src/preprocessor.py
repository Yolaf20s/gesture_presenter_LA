"""
Class Preprocessor: tien xu ly frame.

Strategy cho LA project:
- Detection: resize nho + RGB (MediaPipe chay nhanh)
- Display:   giu nguyen 1280x720 BGR goc (sharp, khong blur)

Khac voi phien ban CV cu (apply blur + resize chung) -> giu chat luong hinh anh.
"""

import cv2
import config


class Preprocessor:
    """Preprocessing: resize+RGB for fast MediaPipe, original BGR for sharp display."""

    def __init__(self, target_size=None):
        if target_size is None:
            target_size = (config.RESIZE_WIDTH, config.RESIZE_HEIGHT)
        self.target_size = target_size
        print(f"[Preprocessor] Detect at {target_size}, display at original resolution.")

    def process(self, frame):
        """
        frame: numpy BGR shape (H, W, 3) goc (vd: 720x1280)
        Return: tuple (rgb_small, bgr_full)
            - rgb_small: resized RGB cho MediaPipe (vd: 360x640)
            - bgr_full:  original BGR cho hien thi (sharp, khong blur)
        """
        # Cho MediaPipe: nho + RGB (khong blur, MediaPipe tu xu ly noise)
        small_bgr = cv2.resize(frame, self.target_size)
        rgb_small = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

        # Cho display: frame goc, khong dung gi
        bgr_full = frame.copy()

        return rgb_small, bgr_full


# ============================================================
# Test
# ============================================================
if __name__ == '__main__':
    """Test: hien original sharp + resized for detection."""
    from src.camera import Camera

    cam = Camera()
    pre = Preprocessor()

    print("Nhan ESC de thoat.")
    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        rgb_small, bgr_full = pre.process(frame)

        cv2.imshow('Display (original sharp)', bgr_full)
        cv2.imshow('Detection (resized RGB)', cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()