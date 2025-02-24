import cv2
import numpy as np

def apply_blur(image: np.ndarray, blur_level: float) -> np.ndarray:
    """
    Apply a blur effect to the input image with a specified level.

    Parameters:
    - image: np.ndarray. The input image to be blurred.
    - blur_level: float. Blur level between 0.0 (no blur) and 1.0 (maximum blur).

    Returns:
    - np.ndarray. The blurred image.
    """
    if not (0.0 <= blur_level <= 1.0):
        raise ValueError("Blur level must be between 0.0 and 1.0.")

    if blur_level == 0.0:
        return image  # No blurring

    # Define kernel size based on blur level
    max_kernel_size = 35  # Maximum blur strength
    kernel_size = int(blur_level * max_kernel_size) | 1  # Ensure it's an odd number

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image
