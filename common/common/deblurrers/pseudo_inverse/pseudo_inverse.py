import numpy as np

from typing import Tuple
from ..classical_deblurrer import ClassicalDeblurrer

class PseudoInverseDeblurrer(ClassicalDeblurrer):
    def __init__(self, alpha: float = 1e-2):
        super().__init__(name="pseudo_inverse")
        self.alpha = alpha

    def _calculate_filter(self, image_dim: Tuple[int], kernel: np.ndarray) -> np.ndarray:
        height, width = image_dim

        kernel = kernel / np.sum(kernel)
        kernel_padded = np.zeros((height, width), dtype=np.float64)
        kh, kw = kernel.shape
        kernel_padded[:kh, :kw] = kernel 

        H = np.fft.fft2(kernel_padded)
        H_conj = np.conj(H)
        H_mag = np.abs(H)
        H_max = np.max(H_mag)

        epsilon = self.alpha * H_max

        mask = H_mag > epsilon

        filter_fft = np.zeros_like(H, dtype=complex)
        filter_fft[mask] = H_conj[mask] / (H_mag[mask] ** 2)

        return filter_fft