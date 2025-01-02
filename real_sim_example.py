import cv2
import numpy as np
from PIL import Image
from utils.dino_sam import get_segmentation
from pose_estimation.pose_estimation_client import get_pose
LABELS = ["can", "table"]
KNOW_LABELS = ["can"]
class RealSimExample:
    def __init__(self):
        self.task_promot = ""
        self.detection_promot = ""
        self.server_url = "http://10.184.17.177:8000/"
        self.pose_url_suffix = "act"
        self.labels = LABELS
        self.camera_intrinsics = np.array([
            [978.735788085938, 0.0, 1030.94287109375],
            [0.0, 979.0402221679688, 766.4556274414062],
            [0.0, 0.0, 1.0]])

    def read_image(self, rgb_image_path, depth_image_path):
        rgb_image = Image.open(rgb_image_path).convert("RGB")
        depth_image = np.load(depth_image_path)
        return rgb_image, depth_image

    def obj_pose_estimation(self, rgb_image, depth_image, mask_image, label):
        url = self.server_url + self.pose_url_suffix
        obj_pose = get_pose(url, rgb_image, depth_image, mask_image, label, self.camera_intrinsics)
        return obj_pose

    def scene_segmentation(self, rgb_image):
        scene_dict = get_segmentation(rgb_image, self.labels, show_segmentation=False)
        return scene_dict

    def scene_vlm_detection(self, rgb_image):
        pass

    def create_simulation(self, scene_dict):
        pass

    def run(self, rgb_image, depth_image):
        scene_dict = self.scene_segmentation(rgb_image)
        scene_dict["poses"] = []
        for i, label in enumerate(scene_dict["labels"]):
            if label == "table":
                scene_dict["poses"].append(np.eye(4))
                continue
            object_mask = scene_dict["masks"][i]
            object_mask_uint8 = (object_mask).astype(np.uint8)
            object_pose = self.obj_pose_estimation(rgb_image, depth_image, object_mask_uint8, label)
            if object_pose is not None:
                scene_dict["poses"].append(object_pose)
            else:
                print("unknown object")
                #TODO: handle unknown object
                scene_dict["poses"].append(np.eye(4))
        return scene_dict

if __name__ == "__main__":
    example = RealSimExample()
    rgb_image_path = "/home/haowen/hw_mine/Real_Sim_Real/test/test_data/scene_1.jpg"
    depth_image_path = "/home/haowen/hw_mine/Real_Sim_Real/test/test_data/scene_1.npy"
    rgb_image, depth_image = example.read_image(rgb_image_path, depth_image_path)
    scene_dict = example.run(rgb_image, depth_image)
    print(scene_dict["poses"])