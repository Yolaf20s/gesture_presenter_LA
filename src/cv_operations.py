"""
Classic computer-vision operations used by the visualizer.

These steps are educational / diagnostic. The live runtime only uses the
resized RGB path in `Preprocessor` for actual inference.
"""

import cv2

import config


class Resizer:
    """Resize to the configured detection resolution."""

    def __init__(self, target_size=None):
        self.target_size = target_size or (config.RESIZE_WIDTH, config.RESIZE_HEIGHT)

    def process(self, frame):
        return cv2.resize(frame, self.target_size)


class Grayscale:
    """Convert BGR to grayscale."""

    def process(self, bgr_frame):
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)


class GaussianFilter:
    """Apply Gaussian blur."""

    def __init__(self, kernel_size=None, sigma=0):
        kernel_size = kernel_size or config.GAUSSIAN_KERNEL
        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma

    def process(self, frame):
        return cv2.GaussianBlur(
            frame,
            (self.kernel_size, self.kernel_size),
            sigmaX=self.sigma,
        )

    def get_kernel(self):
        k1d = cv2.getGaussianKernel(self.kernel_size, self.sigma)
        return k1d @ k1d.T


class EdgeDetector:
    """Apply Canny edge detection."""

    def __init__(self, low_threshold=50, high_threshold=150):
        self.low = low_threshold
        self.high = high_threshold

    def process(self, gray_frame):
        return cv2.Canny(gray_frame, self.low, self.high)


class ColorConverter:
    """Convert BGR to RGB."""

    def process(self, bgr_frame):
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    from src.camera import Camera

    cam = Camera()
    resizer = Resizer()
    grayscale = Grayscale()
    gaussian = GaussianFilter()
    edge = EdgeDetector()

    print("Showing 5 windows: Original / Resized / Gray / Blurred / Edges")
    print("Press ESC to quit.")

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        resized = resizer.process(frame)
        gray = grayscale.process(resized)
        blurred = gaussian.process(gray)
        edges = edge.process(blurred)

        cv2.imshow("1. Original", frame)
        cv2.imshow("2. Resized", resized)
        cv2.imshow("3. Grayscale", gray)
        cv2.imshow("4. Gaussian Blur", blurred)
        cv2.imshow("5. Canny Edges", edges)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
