import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(quaternion, translation):
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
    T_base_eef = quaternion_to_matrix(q_base_eef, t_base_eef)
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
