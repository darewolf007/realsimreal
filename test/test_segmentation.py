from utils.dino_sam import get_segmentation, plot_detections
from pose_estimation.pose_estimation_client import get_pose
import cv2
import numpy as np
from PIL import Image
LABELS = ["table", "can", "cup"]

def test_get_segmentation():
    rgb_image = Image.open("/home/haowen/hw_mine/Real_Sim_Real/data/can_pour_1/3/scene_rgb_image/scene_1.jpg").convert("RGB")
    np_rgb_image = np.array(rgb_image)
    scene_result_dict = get_segmentation(rgb_image, LABELS, show_segmentation=True)
    print(scene_result_dict["labels"])
    for i, label in enumerate(scene_result_dict["labels"]):
        mask = scene_result_dict["masks"][i]
        mask_uint8 = (mask).astype(np.uint8)
        cv2.imwrite(f"test_data/{label}_{i}.png", mask_uint8)

if __name__ == "__main__":
    test_get_segmentation()
    print("test_get_segmentation passed")