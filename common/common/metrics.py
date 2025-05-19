import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import convolve
import torchvision.transforms as transforms
import lpips
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models")

def calculate_ssim(
    img1_np: np.ndarray,
    img2_np: np.ndarray,
    multichannel: bool = True,
    data_range: float = None
    ) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
    img1_np: First image as a numpy array (H, W, C) or (H, W).
    img2_np: Second image as a numpy array (H, W, C) or (H, W).
    multichannel: Whether the images have multiple channels.
    data_range: The data range of the input images (distance between minimum and maximum possible values). 
            If None, it is determined from the image dtype.

    Returns:
    float: SSIM score (1 means identical).
    """
    if data_range is None:
        dmin = min(img1_np.min(), img2_np.min())
        dmax = max(img1_np.max(), img2_np.max())
        data_range = dmax - dmin

    return ssim(img1_np, img2_np, data_range=data_range, channel_axis=-1 if multichannel else None)

def calculate_psnr(
    img1_np: np.ndarray,
    img2_np: np.ndarray,
    data_range: float = None,
    ) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1_np: First image as a numpy array.
        img2_np: Second image as a numpy array.
        data_range: The data range of the input images. If None, it is determined from the image dtype.

    Returns:
        float: PSNR value in dB (higher is better).
    """
    if data_range is None:
        dmin = min(img1_np.min(), img2_np.min())
        dmax = max(img1_np.max(), img2_np.max())
        data_range = dmax - dmin

    mse = np.mean((img1_np.astype(np.float64) - img2_np.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr

def calculate_lpips(
        img1_np: np.ndarray, 
        img2_np: np.ndarray, 
        net_type: str ='alex', 
        device: str ='cuda' if torch.cuda.is_available() else 'cpu'
    ):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS) between two images.

    Args:
        img1_np: First image as a numpy array (H, W, C), RGB, range [0,255] or [0,1].
        img2_np: Second image as a numpy array (H, W, C), RGB, range [0,255] or [0,1].
        net_type: Network type for LPIPS ('alex', 'vgg', or 'squeeze').
        device: Device to run the computation on.

    Returns:
        float: LPIPS score (lower means more similar).
    """
    # Ensure float32 and range [0,1]
    img1_np = img1_np.astype(np.float32)
    img2_np = img2_np.astype(np.float32)
    img1_np = img1_np / 255.0 if img1_np.max() > 1.0 else img1_np
    img2_np = img2_np / 255.0 if img2_np.max() > 1.0 else img2_np

    # Transform to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts HWC [0,1] to CHW [0,1]
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img1_tensor = transform(img1_np).unsqueeze(0).to(device)
    img2_tensor = transform(img2_np).unsqueeze(0).to(device)

    # Load LPIPS model
    loss_fn = lpips.LPIPS(net=net_type, verbose=False).to(device)
    with torch.no_grad():
        lpips_score = loss_fn(img1_tensor, img2_tensor).item()
    return lpips_score

def calculate_gmsd(
    img1_np: np.ndarray,
    img2_np: np.ndarray,
    data_range: float = None
    ) -> float:
    """
    Calculate Gradient Magnitude Similarity Deviation (GMSD) between two images.

    Args:
        img1_np: First image as a numpy array (H, W, C) or (H, W).
        img2_np: Second image as a numpy array (H, W, C) or (H, W).
        data_range: The data range of the input images. If None, it is determined from the image dtype.

    Returns:
        float: GMSD score (lower means more similar).
    """

    def to_gray(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return np.dot(img[..., :3], [0.299, 0.587, 0.114])
        return img

    if data_range is None:
        dmin = min(img1_np.min(), img2_np.min())
        dmax = max(img1_np.max(), img2_np.max())
        data_range = dmax - dmin

    img1 = to_gray(img1_np.astype(np.float32))
    img2 = to_gray(img2_np.astype(np.float32))

    # Normalize to [0, 1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-12)
    img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-12)

    # Prewitt filters for gradient computation
    fx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0
    fy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) / 3.0

    grad1_x = convolve(img1, fx, mode='nearest')
    grad1_y = convolve(img1, fy, mode='nearest')
    grad2_x = convolve(img2, fx, mode='nearest')
    grad2_y = convolve(img2, fy, mode='nearest')

    grad_mag1 = np.sqrt(grad1_x ** 2 + grad1_y ** 2)
    grad_mag2 = np.sqrt(grad2_x ** 2 + grad2_y ** 2)

    T = 0.0026  # recommended by GMSD paper
    gms_map = (2 * grad_mag1 * grad_mag2 + T) / (grad_mag1 ** 2 + grad_mag2 ** 2 + T)
    gmsd = np.std(gms_map)
    return gmsd