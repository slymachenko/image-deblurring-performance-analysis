import cv2
import numpy as np

def compute_rms_contrast(image: np.ndarray) -> float:
    """
    Calculates Root Mean Square (RMS) contrast for a grayscale image.

    Args:
        image: Grayscale image as NumPy array

    Returns:
        float: RMS contrast value
    """
    img_float = image.astype(np.float32) / 255.0
    mean_intensity = np.mean(img_float)
    contrast = np.sqrt(np.mean((img_float - mean_intensity) ** 2))
    return contrast


def compute_sobel_edge_strength(
    image: np.ndarray,
    gaussian_params: dict,
    sobel_params: dict
) -> float:
    """
    Calculates average gradient magnitude using Sobel filters.

    Args:
        image: Grayscale image as NumPy array
        gaussian_params: Gaussian blur settings
        sobel_params: Sobel kernel size

    Returns:
        float: Average gradient (edge strength)
    """
    img_blur = cv2.GaussianBlur(
        image,
        gaussian_params['ksize'],
        sigmaX=gaussian_params['sigma']
    )

    sobel_x = cv2.Sobel(
        img_blur,
        cv2.CV_64F,
        1, 0,
        ksize=sobel_params['ksize']
    )

    sobel_y = cv2.Sobel(
        img_blur,
        cv2.CV_64F,
        0, 1,
        ksize=sobel_params['ksize']
    )

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradient_magnitude, np.mean(gradient_magnitude)


def compute_canny_edge_density(
        image: np.ndarray, 
        gaussian_params: dict, 
        canny_thresholds: dict = None
) -> float:
    """
    Calculates edge density as the proportion of edge pixels using Canny edge detection.

    Args:
        image: Grayscale image as NumPy array (uint8)
        gaussian_params: Dictionary with 'ksize' (tuple) and 'sigma' (float) for Gaussian blur
        canny_thresholds (optional): Dictionary with 'low' and 'high' thresholds for Canny edge detection

    Returns:
        float: Edge density (proportion of edge pixels, in [0, 1])

    Raises:
        ValueError: If input image is invalid or parameters are incorrect
    """
    # Validate input
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 2:
        raise ValueError("Input must be a valid grayscale image (2D NumPy array)")

    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(
        image,
        gaussian_params['ksize'],
        sigmaX=gaussian_params['sigma']
    )

    # Calculate Canny thresholds if not provided
    if canny_thresholds is None:
        img_median = np.median(img_blur)
        img_sigma = 0.33
        canny_thresholds = {
            'low': int(max(0, (1.0 - img_sigma) * img_median)),
            'high': int(min(255, (1.0 + img_sigma) * img_median))
        }

    # Apply Canny edge detection
    edges = cv2.Canny(
        img_blur,
        canny_thresholds['low'],
        canny_thresholds['high']
    )

    # Compute edge density as proportion of edge pixels
    edge_density = np.sum(edges > 0) / edges.size

    return edges, edge_density