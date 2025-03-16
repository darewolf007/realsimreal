import os
import cv2
import pickle
import numpy as np

def resize_image(image, scale_factor=0.5):                
    height, width = image.shape[:2]
    min_dim = min(height, width)
    start_y = (height - min_dim) // 2
    start_x = (width - min_dim) // 2
    cropped = image[start_y:start_y+min_dim, start_x:start_x+min_dim]
    new_size = (int(min_dim * scale_factor), int(min_dim * scale_factor))
    resized_image = cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)
    return resized_image

def save_image_pkl(image_dict, path, save_ori_image = False):
    if not os.path.exists(path):
         os.makedirs(path)
    pkl_num = sum(1 for name in os.listdir(path) if name.endswith('.pkl'))
    pkl_path = os.path.join(path, f"{pkl_num + 1}.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(image_dict, file)
    if save_ori_image:
        for key, image in image_dict.items():
            if 'view' in key:
                save_path = os.path.join(path, f"{key}_{pkl_num + 1}.png")
                cv2.imwrite(save_path, image)