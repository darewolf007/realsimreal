import open3d as o3d
import numpy as np
import cv2
from PIL import Image
from simple_sim.unknow_obj.fit_unknow_obj import process_unknow_obj

def get_handeye_T(file_path):
    import yaml
    from scipy.spatial.transform import Rotation as R
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    qw = data['transformation']['qw']
    qx = data['transformation']['qx']
    qy = data['transformation']['qy']
    qz = data['transformation']['qz']
    x = data['transformation']['x']
    y = data['transformation']['y']
    z = data['transformation']['z']
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [x, y, z]
    return transformation_matrix

def test_process_unknow_obj():
    rgb_image = Image.open("/home/haowen/hw_mine/Real_Sim_Real/data/can_pour_1/3/scene_rgb_image/scene_1.jpg").convert("RGB")
    depth_image = np.load("/home/haowen/hw_mine/Real_Sim_Real/data/can_pour_1/3/scene_depth_image/scene_1.npy")
    table_mask_image = cv2.imread("/home/haowen/hw_mine/Real_Sim_Real/test/test_data/table._1.png", cv2.IMREAD_GRAYSCALE)
    other_mask_image1 = cv2.imread("/home/haowen/hw_mine/Real_Sim_Real/test/test_data/can._2.png", cv2.IMREAD_GRAYSCALE)
    other_mask_image2 = cv2.imread("/home/haowen/hw_mine/Real_Sim_Real/test/test_data/cup._0.png", cv2.IMREAD_GRAYSCALE)
    intrinsic = np.array([
        [978.735788085938, 0.0, 1030.94287109375],
        [0.0, 979.0402221679688, 766.4556274414062],
        [0.0, 0.0, 1.0]])
    handeye_T = get_handeye_T("/home/haowen/hw_mine/Real_Sim_Real/data/can_pour_1/ur5_kinect_handeyecalibration_eye_on_base.yaml")
    obj_hull = process_unknow_obj(rgb_image, depth_image, table_mask_image, intrinsic,label='table', other_mask=[other_mask_image2, other_mask_image1], filter=False, depth_scale=1.0, handeye_T=handeye_T, render=False)
if __name__ == "__main__":
    test_process_unknow_obj()
    print("test_process_unknow_obj passed")