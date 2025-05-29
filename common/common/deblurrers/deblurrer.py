from abc import ABC, abstractmethod
import numpy as np

class Deblurrer(ABC):
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        """Return the name of the deblurrer instance."""
        return self.name

    @abstractmethod
    def deblur(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur the input image.

        Parameters:
        image (np.ndarray): Input image (grayscale or color, BGR format).

        Returns:
        np.ndarray: Deblurred image in BGR format.
        """
        pass