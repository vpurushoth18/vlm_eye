import cv2
import numpy as np
import os
from tqdm import tqdm

# Function to add snow effect
def add_snow(image, intensity=0.5):
    """
    Adds a snow effect to an image.
    intensity: Controls the amount of snow (0 = no snow, 1 = maximum snow)
    """
    h, w, _ = image.shape
    snow_layer = np.zeros((h, w), dtype=np.uint8)

    # Number of snowflakes based on intensity
    num_snowflakes = int((h * w) * intensity * 0.005)

    for _ in range(num_snowflakes):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        size = np.random.randint(1, 4)  # Small snowflakes
        cv2.circle(snow_layer, (x, y), size, 255, -1)

    # Blend the snow layer with the image
    snow_image = cv2.addWeighted(image, 1 - intensity, cv2.cvtColor(snow_layer, cv2.COLOR_GRAY2BGR), intensity, 0)
    return snow_image

# Main function to process images
def process_images(input_folder, output_folder, snow_levels=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for level in tqdm(range(1, snow_levels + 1)):
        snow_intensity = level * 0.2  # Increasing snow level gradually
        level_folder = os.path.join(output_folder, f"Snow_Level_{int(snow_intensity * 100)}")
        os.makedirs(level_folder, exist_ok=True)

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)

            # Apply snow effect
            snowy_image = add_snow(image, intensity=snow_intensity)

            output_path = os.path.join(level_folder, image_file)
            cv2.imwrite(output_path, snowy_image)
