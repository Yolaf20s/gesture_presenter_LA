"""
Cac phep toan Computer Vision co dien — tach ra de hien thi rieng tung step.

Linear Algebra dung trong file nay:
- Grayscale: weighted sum cua 3 kenh RGB (linear combination)
- Gaussian filter: convolution voi kernel symmetric (linear operator)
- Sobel/Canny: gradient = phep toan tuyen tinh tren neighborhood

Cac class:
- Resizer         — Step 2a: resize giam tai
- Grayscale       — Step 2b: BGR -> Gray (1 kenh)
- GaussianFilter  — Step 2c: smooth voi Gaussian kernel
- EdgeDetector    — Step 2d: Canny edge detection (chi de visualize)
- ColorConverter  — Step 2e: BGR -> RGB cho MediaPipe
"""

import cv2
import numpy as np
import config


class Resizer:
    """Step 2a: Resize giam tai tinh toan."""

    def __init__(self, target_size=None):
        if target_size is None:
            target_size = (config.RESIZE_WIDTH, config.RESIZE_HEIGHT)
        self.target_size = target_size

    def process(self, frame):
        return cv2.resize(frame, self.target_size)


class Grayscale:
    """Step 2b: Chuyen BGR sang grayscale.

    Linear Algebra: gray = 0.299*R + 0.587*G + 0.114*B
    Day la weighted sum (linear combination) cua 3 kenh.
    Cong thuc nay theo standard luma coefficients (ITU-R BT.601).
    """

    def process(self, bgr_frame):
        """bgr_frame: shape (H, W, 3). Return: gray (H, W)."""
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)


class GaussianFilter:
    """Step 2c: Lam mo voi Gaussian kernel.

    Linear Algebra: convolution voi kernel Gaussian symmetric.
    Output[i,j] = sum_uv kernel[u,v] * Input[i+u, j+v]
    Day la phep toan TUYEN TINH (linear operator).
    Kernel Gaussian la separable -> co the tach thanh 2 1D conv (toi uu).
    """

    def __init__(self, kernel_size=None, sigma=0):
        if kernel_size is None:
            kernel_size = config.GAUSSIAN_KERNEL
        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size phai la so le, nhan {kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma  # 0 = tu tinh tu kernel size

    def process(self, frame):
        """frame: gray hoac BGR deu duoc. Return: blurred frame cung shape."""
        return cv2.GaussianBlur(
            frame,
            (self.kernel_size, self.kernel_size),
            sigmaX=self.sigma,
        )

    def get_kernel(self):
        """Tra ve Gaussian kernel 2D (de visualize trong report).
        Day la ma tran symmetric, all positive, sum = 1.
        """
        k1d = cv2.getGaussianKernel(self.kernel_size, self.sigma)
        return k1d @ k1d.T  # outer product -> 2D kernel


class EdgeDetector:
    """Step 2d: Canny edge detection.

    KHONG dung cho classification (ban tay/co the can pixel mau).
    Chi dung de visualize, chung minh co Canny trong pipeline.
    """

    def __init__(self, low_threshold=50, high_threshold=150):
        self.low = low_threshold
        self.high = high_threshold

    def process(self, gray_frame):
        """gray_frame: 1 kenh. Return: edge map (binary)."""
        return cv2.Canny(gray_frame, self.low, self.high)


class ColorConverter:
    """Step 2e: BGR -> RGB cho MediaPipe."""

    def process(self, bgr_frame):
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)


# ============================================================
# Test khi chay file truc tiep
# ============================================================
if __name__ == '__main__':
    """Test cac CV operation tren 1 frame: hien 5 cua so."""
    from src.camera import Camera

    cam = Camera()

    resizer = Resizer()
    grayscale = Grayscale()
    gaussian = GaussianFilter()
    edge = EdgeDetector()

    print("Hien 5 cua so: Original / Resized / Gray / Blurred / Edge")
    print("Nhan ESC de thoat.")

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        resized = resizer.process(frame)
        gray = grayscale.process(resized)
        blurred = gaussian.process(gray)
        edges = edge.process(blurred)

        cv2.imshow('1. Original (1280x720)', frame)
        cv2.imshow('2. Resized (640x360)', resized)
        cv2.imshow('3. Grayscale', gray)
        cv2.imshow('4. Gaussian Blur', blurred)
        cv2.imshow('5. Canny Edges', edges)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()