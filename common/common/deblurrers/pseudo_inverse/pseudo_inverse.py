import numpy as np
import cv2
from ..base import Deblurrer

class PseudoInverseDeblurrer(Deblurrer):
    def __init__(self, regularization_lambda: float = 1e-3, kernel_size: int = 10):
        self.name = "pseudo_inverse"
        self.regularization_lambda = regularization_lambda
        self.kernel_size = kernel_size

    def get_name(self) -> str:
        return self.name

    def _apply_to_channel(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # convert image data into float type for precise math
        image = image.astype(float)
        
        # pad the kernel to match the image size; kernel is normalized to sum=1
        kernel_padded = np.zeros_like(image)
        kh, kw = kernel.shape
        kernel_padded[:kh, :kw] = kernel / np.sum(kernel)
        
        # compute the FFT for both image and padded kernel
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel_padded)
        
        # compute the conjugate and squared absolute value of the kernel FFT
        kernel_fft_conj = np.conj(kernel_fft)
        kernel_fft_abs_sq = np.abs(kernel_fft) ** 2
        
        # compute the pseudo-inverse using regularization (to avoid division by zero)
        pseudo_inverse = kernel_fft_conj / (kernel_fft_abs_sq + self.regularization_lambda)
        
        # apply pseudo-inverse in the frequency domain
        deblurred_fft = image_fft * pseudo_inverse
        
        # transform back to spatial domain and ensure only the real part is used
        deblurred = np.fft.ifft2(deblurred_fft).real
        
        # clip pixel values to be in valid range
        return np.clip(deblurred, 0, 255)

    def _estimate_kernel(self, image: np.ndarray) -> np.ndarray:
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