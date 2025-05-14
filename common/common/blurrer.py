import numpy as np
import cv2

def gen_gaussian_kernel(size, sigma):
    """Generate a Gaussian blur kernel."""
    k1d = cv2.getGaussianKernel(size, sigma)
    kernel = k1d @ k1d.T
    return kernel

def gen_motion_kernel(size, length, angle):
    """Generate a motion blur kernel based on length and angle."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2

    angle_rad = np.deg2rad(angle)
    dx = length / 2 * np.cos(angle_rad)
    dy = length / 2 * np.sin(angle_rad)

    start_point = (int(center - dx), int(center - dy))
    end_point = (int(center + dx), int(center + dy))
    
    cv2.line(kernel, start_point, end_point, 1, 1)
    kernel /= np.sum(kernel)
    return kernel

def gen_box_kernel(size):
    """Generate a box blur kernel."""
    return np.ones((size, size), dtype=np.float32) / (size * size)