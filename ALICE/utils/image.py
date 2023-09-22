import cv2
import numpy as np
from pathlib import Path


def resize_image(image, max_width, max_height):

    # Get the original image dimensions
    height, width = image.shape[:2]
    ratio = []

    if height > max_height:
        ratio.append(max_height / height)

    if width > max_width:
        ratio.append(max_width / width)

    # If image needs to be resized, select smallest ratio (0.2 etc.,)
    # and resize both sides to that ratio
    if ratio:
        r = min(ratio)        
        new_width = round(r * width)
        new_height = round(r * height)
        image = cv2.resize(image, (new_width, new_height))

    return image

def save_image(image, path: Path):
    cv2.imwrite(str(path), image) 
    
def overlay_image(image1, image2):
    """
    Overlay an image over another with 50% opacity
    """
    overlay = image1.copy()    
    opacity = 0.5
    return cv2.addWeighted(image2, opacity, overlay, 1 - opacity, 0, overlay)   

def pad_image_to_width(image, target_width, pad_color=(255, 255, 255)):
    """
    Pad an image to the target width with a specified color
    """
    current_width = image.shape[1]


    if current_width > target_width:
        raise Exception('Image exceeds target width')
    elif current_width == target_width:
        return image

    # # Create a new image with the target width and fill it with the pad_color
    padded_image = np.zeros((image.shape[0], target_width, 3), dtype=np.uint8)    
    padded_image[:, :] = pad_color

    # Copy the original image to the left side of the padded image
    padded_image[:, :current_width] = image

    return padded_image
