import numpy as np

from common.img_features import compute_sobel_edge_strength
from common.config import SOBEL_PARAMS

from .deblurrer import Deblurrer

class ClassicalDeblurrer(Deblurrer):
    def __init__(self, name: str):
        super().__init__(name)
    
    def _estimate_kernel(self, image: np.ndarray) -> np.ndarray:
        # Compute the gradient magnitude using the existing function.
        gradient_magnitude, _ = compute_sobel_edge_strength(
            image=image, 
            gaussian_params = {
                "ksize": (1, 1),
                "sigma": 0.0
            }, 
            sobel_params=SOBEL_PARAMS
        )
        
        # Threshold the gradient to focus on strong edges.
        threshold = 0.2 * gradient_magnitude.max()
        edge_map = np.where(gradient_magnitude >= threshold, gradient_magnitude, 0)
        
        # Extract horizontal and vertical edge profiles.
        x_profile = np.mean(edge_map, axis=0)
        y_profile = np.mean(edge_map, axis=1)
        
        # Define a heuristic kernel size.
        kernel_size = 7
        center_x = len(x_profile) // 2
        center_y = len(y_profile) // 2
        half_size = kernel_size // 2
        
        # Extract the central window for the kernel from the edge profiles.
        x_kernel = x_profile[max(0, center_x - half_size):center_x + half_size + 1]
        y_kernel = y_profile[max(0, center_y - half_size):center_y + half_size + 1]
        
        # Pad the profiles if the extracted window is smaller than the desired kernel size.
        if len(x_kernel) < kernel_size:
            pad_width = kernel_size - len(x_kernel)
            x_kernel = np.pad(x_kernel, (pad_width // 2, pad_width - pad_width // 2), mode='edge')
        if len(y_kernel) < kernel_size:
            pad_width = kernel_size - len(y_kernel)
            y_kernel = np.pad(y_kernel, (pad_width // 2, pad_width - pad_width // 2), mode='edge')
        
        # Form a separable kernel as the outer product of the two 1D profiles.
        kernel = np.outer(y_kernel, x_kernel)
        
        # Normalize the kernel so that its sum equals 1.
        kernel_sum = kernel.sum()
        if kernel_sum != 0:
            kernel = kernel / kernel_sum
        else:
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        return kernel

    def _apply_to_channel(self, image: np.ndarray, H: np.ndarray) -> np.ndarray:
        # G    : blurred image frequency representation
        # H    : kernel frequency representation
        # F_hat: deblurred image frequency representation

        G = np.fft.fft2(image)
        F_hat = G * H
        return np.real(np.fft.ifft2(F_hat))

    def deblur(self, image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
        image = image.astype(np.float64)
    
        if image.ndim == 2:
            image = image[:, :, None]
        height, width, channels = image.shape

        if kernel is None:
            kernel = self._estimate_kernel(image[:, :, 0])
        
        kernel = kernel / np.sum(kernel)

        filter_fft = self._calculate_filter((height, width), kernel)

        deblurred = np.empty_like(image, dtype=np.float64)
        for ch in range(channels):
            deblurred[:, :, ch] = self._apply_to_channel(image[:, :, ch], filter_fft)
        
        return np.clip(deblurred.squeeze(), 0, 255).astype(np.uint8)