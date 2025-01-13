import cv2
import numpy as np

def resize_image(image, scale_factor=0.5):                
    height, width = image.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image