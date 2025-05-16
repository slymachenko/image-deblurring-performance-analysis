import numpy as np
import cv2

def gen_gaussian_kernel(size, sigma):
    """Generate a Gaussian blur kernel."""
    k1d = cv2.getGaussianKernel(size, sigma)
    kernel = k1d @ k1d.T
    return kernel

def gen_motion_kernel(length, angle):
    """Generate a motion blur kernel based on length and angle."""
    if length % 2 == 0:
        length += 1

    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0

    angle_rad = np.deg2rad(angle)
    dx = length / 2.0 * np.cos(angle_rad)
    dy = length / 2.0 * np.sin(angle_rad)

    start_point = (int(center - dx), int(center - dy))
    end_point = (int(center + dx), int(center + dy))
    
    cv2.line(kernel, start_point, end_point, 1, 1, lineType=cv2.LINE_AA)
    kernel /= np.sum(kernel)
    return kernel

def gen_box_kernel(size):
    """Generate a box blur kernel."""
    return np.ones((size, size), dtype=np.float32) / (size * size)