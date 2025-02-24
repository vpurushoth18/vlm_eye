import cv2
import numpy as np
import random

def add_haze(image, haze_level=0.5):
    """
    Add haze noise to an image. The haze_level controls the intensity of the haze.

    Parameters:
        image (numpy array): The input image to which haze will be added.
        haze_level (float): The level of haze noise to be added (0.0 to 1.0).

    Returns:
        numpy array: The hazy image.
    """

    image = image.astype(np.float32) / 255.0

    haze_mask = np.random.uniform(0, 1, image.shape).astype(np.float32)

    hazy_image = image * (1 - haze_level) + haze_mask * haze_level

    hazy_image = np.clip(hazy_image * 255.0, 0, 255).astype(np.uint8)

    return hazy_image
