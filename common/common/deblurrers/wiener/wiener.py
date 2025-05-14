import numpy as np
import cv2
from ..base import Deblurrer

class WienerDeblurrer(Deblurrer):
    def __init__(self, K: float = 1e-3, kernel_size: int = 10):
        self.name = "Wiener"
        self.K = K
        self.kernel_size = kernel_size

    def get_name(self) -> str:
        return self.name

    def _apply_to_channel(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        image = image.astype(float)
        kernel_padded = np.zeros_like(image)
        kh, kw = kernel.shape
        kernel_padded[:kh, :kw] = kernel / np.sum(kernel)
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel_padded)
        kernel_fft_conj = np.conj(kernel_fft)
        kernel_fft_abs_sq = np.abs(kernel_fft) ** 2
        wiener_filter = kernel_fft_conj / (kernel_fft_abs_sq + self.K)
        deblurred_fft = image_fft * wiener_filter
        deblurred = np.fft.ifft2(deblurred_fft).real
        return np.clip(deblurred, 0, 255)

    def _estimate_kernel(self, image: np.ndarray) -> np.ndarray:
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        thresh = np.percentile(grad_magnitude, 95)
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        kernel[center, center] = 1.0
        for i in range(-self.kernel_size//2 + 1, self.kernel_size//2):
            for j in range(-self.kernel_size//2 + 1, self.kernel_size//2):
                if i != 0 or j != 0:
                    kernel[center + i, center + j] = np.exp(-(i**2 + j**2) / (2 * (self.kernel_size / 4)**2))
        return kernel / np.sum(kernel)

    def deblur(self, image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            result = np.zeros_like(image, dtype=float)
            for channel in range(3):
                if kernel is None:
                    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
                    kernel = self._estimate_kernel(gray)
                result[:, :, channel] = self._apply_to_channel(image[:, :, channel], kernel)
        else:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if kernel is None:
                kernel = self._estimate_kernel(image[:, :, 0])
            result = np.zeros_like(image, dtype=float)
            for channel in range(3):
                result[:, :, channel] = self._apply_to_channel(image[:, :, channel], kernel)
        return result.astype(np.uint8)