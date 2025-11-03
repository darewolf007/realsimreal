import os
import yaml
import cv2
import numpy as np
from PIL import Image
from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator

def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, 'configs/config.yaml')
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def show_img(rgb):
    cv2.imshow('img', rgb[..., ::-1])
    cv2.waitKey(0)
    print('showing image, click on the window and press "ESC" to close and continue')
    cv2.destroyAllWindows()


def depth_map_to_point_cloud(depth_map, intrinsic_matrix):
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError("intrinsic_matrix 3x3")
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_map.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points_xyz = np.stack([X, Y, Z], axis=-1)
    flat_points = points_xyz.reshape(-1, 3)
    # valid_mask = (flat_points[:, 2] > 0)
    # point_cloud = flat_points[valid_mask]
    return points_xyz


def scale_image_and_intrinsics(image_data, intrinsic_matrix, new_width, new_height, is_mask=False):
    if image_data.ndim == 3:
        original_height, original_width, _ = image_data.shape
    else:
        original_height, original_width = image_data.shape
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    if is_mask or image_data.dtype in [np.int8, np.int16, np.int32] or image_data.dtype == np.uint16:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
    scaled_image = cv2.resize(
        image_data,
        (new_width, new_height),
        interpolation=interpolation
    )
    new_K = intrinsic_matrix.copy()
    new_K[0, 0] *= scale_x
    new_K[1, 1] *= scale_y
    new_K[0, 2] *= scale_x
    new_K[1, 2] *= scale_y
    return scaled_image, new_K

if __name__ == "__main__":
    rgb_image_path = "/home/haowen/hw_dataset/test_data/real_single_can/scene_1.jpg"
    depth_image_path = "/home/haowen/hw_dataset/test_data/real_single_can/scene_1.npy"
    mask_image_path = "/home/haowen/hw_dataset/test_data/real_single_can/can.png"
    config_path = "/home/haowen/hw_mine/Real_Sim_Real/affordance/rekep/config.yaml"
    camera_intrinsics = np.array([
        [978.735788085938, 0.0, 1030.94287109375],
        [0.0, 979.0402221679688, 766.4556274414062],
        [0.0, 0.0, 1.0]])
    global_config = get_config(config_path)
    rgb_image = np.array(Image.open(rgb_image_path).convert("RGB"))
    depth_image = np.load(depth_image_path).astype(np.float32)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    ORIGINAL_H, ORIGINAL_W = 1536, 2048
    TARGET_SCALE_FACTOR = 0.25
    TARGET_W = int(ORIGINAL_W * TARGET_SCALE_FACTOR)
    TARGET_H = int(ORIGINAL_H * TARGET_SCALE_FACTOR)
    scaled_rgb_image, new_K = scale_image_and_intrinsics(
        rgb_image,
        camera_intrinsics,
        TARGET_W, TARGET_H,
        is_mask=False
    )
    scaled_depth_image, _ = scale_image_and_intrinsics(
        depth_image,
        camera_intrinsics,  # 使用原始 K 计算缩放，确保缩放因子一致
        TARGET_W, TARGET_H,
        is_mask=False
    )
    scaled_mask_image, _ = scale_image_and_intrinsics(
        mask_image,
        camera_intrinsics,
        TARGET_W, TARGET_H,
        is_mask=True
    )
    points = depth_map_to_point_cloud(scaled_depth_image, new_K)
    keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
    keypoints, projected_img = keypoint_proposer.get_keypoints(scaled_rgb_image, points, scaled_mask_image)
    # points = depth_map_to_point_cloud(depth_image, camera_intrinsics)
    # keypoints, projected_img = keypoint_proposer.get_keypoints(rgb_image, points, mask_image)
    show_img(projected_img)
    instruction = 'reorient the white pen and drop it upright into the black pen holder'
    # metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
    # constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
    # rekep_program_dir = constraint_generator.generate(projected_img, instruction, metadata)