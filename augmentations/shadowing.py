import cv2
import numpy as np
import os
from tqdm import tqdm

# Function to add shadow effect
def add_shadow(image, intensity=0.5):
    """
    Adds a shadow effect to an image.
    intensity: Controls the darkness of the shadow (0 = no shadow, 1 = maximum shadow)
    """
    h, w, _ = image.shape
    shadow_layer = np.zeros_like(image, dtype=np.uint8)

    # Create a gradient shadow mask
    x_start, y_start = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
    x_end, y_end = np.random.randint(w // 2, w), np.random.randint(h // 2, h)
    
    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(shadow_mask, (x_start, y_start), (x_end, y_end), (255,), -1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (101, 101), 0)

    # Normalize the shadow intensity
    shadow_mask = (shadow_mask / 255.0) * intensity
    shadow_layer = np.dstack([shadow_mask] * 3)  # Convert to 3-channel
    
    # Apply shadow effect
    shadow_image = cv2.addWeighted(image.astype(np.float32), 1 - intensity, shadow_layer * 255, intensity, 0)
    return shadow_image.astype(np.uint8)

# Main function to process images
def process_images(input_folder, output_folder, shadow_levels=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for level in tqdm(range(1, shadow_levels + 1)):
        shadow_intensity = level * 0.2  # Increasing shadow level gradually
        level_folder = os.path.join(output_folder, f"Shadow_Level_{int(shadow_intensity * 100)}")
        os.makedirs(level_folder, exist_ok=True)

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)

            # Apply shadow effect
            shadowed_image = add_shadow(image, intensity=shadow_intensity)

            output_path = os.path.join(level_folder, image_file)
            cv2.imwrite(output_path, shadowed_image)

