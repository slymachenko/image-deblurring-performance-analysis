from abc import ABC, abstractmethod
import numpy as np

class Deblurrer(ABC):
    @abstractmethod
    def deblur(self, image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
        """
        Deblur the input image.

        Parameters:
        image (np.ndarray): Input image (grayscale or color, BGR format).
        kernel (np.ndarray, optional): Blur kernel for non-AI methods. If None, estimate it.

        Returns:
        np.ndarray: Deblurred image in BGR format.
        """
        pass