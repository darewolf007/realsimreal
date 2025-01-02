import open3d as o3d
import numpy as np
import cv2
from PIL import Image
from simple_sim.unknow_obj.fit_unknow_obj import process_unknow_obj

def test_process_unknow_obj():
    rgb_image = Image.open("test_data/scene_1.jpg").convert("RGB")
    depth_image = np.load("test_data/scene_1.npy")
    table_mask_image = cv2.imread("test_data/table_1.png", cv2.IMREAD_GRAYSCALE)
    other_mask_image = cv2.imread("test_data/can_0.png", cv2.IMREAD_GRAYSCALE)
    intrinsic = np.array([
        [978.735788085938, 0.0, 1030.94287109375],
        [0.0, 979.0402221679688, 766.4556274414062],
        [0.0, 0.0, 1.0]])
    obj_hull = process_unknow_obj(rgb_image, depth_image, table_mask_image, intrinsic,label='table', other_mask=[other_mask_image], filter=False, depth_scale=1.0)

if __name__ == "__main__":
    test_process_unknow_obj()
    print("test_process_unknow_obj passed")