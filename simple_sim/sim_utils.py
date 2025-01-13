import numpy as np
import os
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def get_handeye_T(file_path):
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

def quaternion_to_matrixT(quaternion, translation):
    rot_matrix = R.from_quat(quaternion).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def matrix_to_translation_quaternion(matrix):
    assert matrix.shape == (4, 4), "Input matrix must be a 4x4 transformation matrix."
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    return translation, quaternion

def transform_to_camera_frame(T_base_camera, q_base_eef, t_base_eef):
    T_base_eef = quaternion_to_matrixT(q_base_eef, t_base_eef)
    T_camera_eef = np.dot(np.linalg.inv(T_base_camera), T_base_eef)
    translation, quaternion = matrix_to_translation_quaternion(T_camera_eef)
    return translation, quaternion

def matrix_to_translation_quaternion(matrix):
    assert matrix.shape == (4, 4), "Input matrix must be a 4x4 transformation matrix."
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    return translation, quaternion

def adjust_orientation_to_z_up(transformation_matrix):
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]
    z_axis = rotation_matrix[:, 2]
    target_z_axis = np.array([0, 0, 1])
    rotation_to_align_z = R.align_vectors([target_z_axis], [z_axis])[0]
    adjusted_rotation_matrix = rotation_to_align_z.as_matrix().dot(rotation_matrix)
    adjusted_transformation_matrix = np.eye(4)
    adjusted_transformation_matrix[:3, :3] = adjusted_rotation_matrix
    adjusted_transformation_matrix[:3, 3] = translation_vector
    return adjusted_transformation_matrix

def residuals(params, sim_data, real_data):
    rotation_vector = params[:3]
    translation_vector = params[3:]
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    transformed_sim_data = sim_data.dot(rotation_matrix.T) + translation_vector
    residuals = transformed_sim_data - real_data
    return residuals.ravel()

def fit_transformation(sim_data, real_data):
    initial_params = np.zeros(6)
    result = least_squares(residuals, initial_params, args=(sim_data, real_data))
    return result

def crop_image(image, center, size):
    x, y = center
    width, height = size
    x1 = max(0, x - width // 2)
    y1 = max(0, y - height // 2)
    x2 = min(image.shape[1], x + width // 2)
    y2 = min(image.shape[0], y + height // 2)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def get_7Dof_pose(matrix_T):
    rotation = R.from_matrix(matrix_T[:3, :3])
    quaternion = rotation.as_quat()
    translation = matrix_T[:3, 3]
    return np.concatenate([translation, quaternion])

def add_noise_to_rotation_z(roation_quat, rotation_noise_bounds):
    rotation = R.from_quat(roation_quat[[1, 2, 3, 0]])
    z_rotation_noise = R.from_euler('z', np.random.uniform(rotation_noise_bounds[0], rotation_noise_bounds[1]), degrees=True)
    noisy_rotation = z_rotation_noise * rotation
    noisy_quaternion_xyzw = noisy_rotation.as_quat()
    noisy_quaternion_wxyz = noisy_quaternion_xyzw[[3, 0, 1, 2]]
    return  noisy_quaternion_wxyz

def count_files_in_directory(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

def test_delta_end(env_info):
    real_data = []
    sim_data = []
    for i in range(1, 150):
        joint_data = np.load(env_info['data_path'] + "joint_" + str(i) + ".npy")
        traj_data = np.load(env_info['data_path'] + "traj_" + str(i) + ".npy")
        print("b",joint_data)
        joint_data[0], joint_data[2] = joint_data[2], joint_data[0]
        print("a",joint_data)
        translation_data = traj_data[:3] + np.array([0.0, 0.0, 0.2])
        save_env_info = env_info.copy()
        save_env_info['robot_init_qpos'] = joint_data
        save_env_info['hand_eye'] = np.eye(4)
        test_real = RealInSimulation("UR5e",
                                save_env_info,
                                has_renderer=False,  # no on-screen renderer
                                has_offscreen_renderer=False,  # no off-screen renderer
                                ignore_done=True,
                                use_camera_obs=False,  # no camera observations
                                control_freq=20,
                                renderer="mjviewer",
                                camera_heights=[1536, 1536, 1536],
                                camera_widths=[2048, 2048, 2048],
                                camera_names=["birdview", "frontview", "rightview"],)
        test_real.reset()
        now_end = test_real.env.sim.data.get_site_xpos('robot0_attachment_site')
        print("gripper2", test_real.env.sim.data.get_site_xpos('gripper0_right_grip_site_cylinder'))
        print("now", now_end)
        print("real", translation_data)
        real_data.append(translation_data)
        sim_data.append(now_end)
    result = fit_transformation(np.array(real_data), np.array(sim_data))
    rotation_vector = result.x[:3]
    translation_vector = result.x[3:]
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    print("Rotation Matrix:\n", rotation_matrix)
    print("Translation Vector:\n", translation_vector)
    return rotation_matrix, translation_vector