import numpy as np

from ..classical_deblurrer import ClassicalDeblurrer

from typing import Tuple

class WienerDeblurrer(ClassicalDeblurrer):
    def __init__(self, reg: float = 1e-3):
        super().__init__(name="wiener")
        self.reg = reg

    def _calculate_filter(self, image_dim: Tuple[int], kernel: np.ndarray) -> np.ndarray:
        height, width = image_dim

        H = np.fft.fft2(kernel, s=(height, width))
        H_conj = np.conj(H)
        H_mag = np.abs(H)

        return H_conj / (H_mag ** 2 + self.reg)