import numpy as np
from scipy.spatial.transform import Rotation as R

def matrix_to_translation_quaternion(matrix):
    assert matrix.shape == (4, 4), "Input matrix must be a 4x4 transformation matrix."
    
    # Extract translation
    translation = matrix[:3, 3]
    
    # Extract rotation matrix
    rotation_matrix = matrix[:3, :3]
    
    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    
    return translation, quaternion

import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_view_matrix(quaternion, position):
    """
    Convert a quaternion and position to a view matrix.
    
    Args:
        quaternion (np.ndarray): A 1x4 array of quaternion (x, y, z, w).
        position (np.ndarray): A 1x3 array of position (x, y, z).
        
    Returns:
        view_matrix (np.ndarray): A 4x4 view matrix.
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    
    # Create a 4x4 identity matrix
    view_matrix = np.eye(4)
    
    # Set the rotation part
    view_matrix[:3, :3] = rotation_matrix.T
    
    # Set the translation part
    view_matrix[:3, 3] = -np.dot(rotation_matrix.T, position)
    
    return view_matrix

def quaternion_to_euler(quaternion):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quaternion (np.ndarray): A 1x4 array of quaternion (x, y, z, w).
        
    Returns:
        euler_angles (np.ndarray): A 1x3 array of Euler angles (roll, pitch, yaw) in radians.
    """
    # Convert quaternion to rotation object
    rotation = R.from_quat(quaternion)
    
    # Convert rotation object to Euler angles
    euler_angles = rotation.as_euler('xyz', degrees=False)
    
    return euler_angles

def translation_quaternion_to_matrix(translation, quaternion):
    """
    Convert translation and quaternion to a 4x4 transformation matrix.
    
    Args:
        translation (np.ndarray): A 1x3 array of translation (x, y, z).
        quaternion (np.ndarray): A 1x4 array of quaternion (x, y, z, w).
        
    Returns:
        matrix (np.ndarray): A 4x4 transformation matrix.
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    
    # Create a 4x4 identity matrix
    matrix = np.eye(4)
    
    # Set the rotation part
    matrix[:3, :3] = rotation_matrix
    
    # Set the translation part
    matrix[:3, 3] = translation
    
    return matrix

# Example usage:
quaternion = [1, 0, 0, 0]  # Identity quaternion
position = [0, 0, 0]  # Camera position

view_matrix = quaternion_to_view_matrix(quaternion, position)
print("View Matrix:\n", view_matrix)
# -0.7629
# 0.8220
# 0.2512
position = np.array([-0.7629, 0.8220, 0.2512])
quaternion = np.array([0.1833, -0.3112, 0.7886, -0.4975])
robot = np.array([-0.3112, 0.7886, -0.4975, 0.1833])
view_matrix = translation_quaternion_to_matrix(position, quaternion)
a = np.array(
    [[-0.71984259, -0.2949338,   0.6283635,  -0.76957957],
 [-0.69408555,  0.29477574, -0.65677432,  0.82485058],
 [ 0.00847863, -0.90891216, -0.41690143,  0.23738398],
 [ 0.,          0. ,         0.,          1.  ,      ]]
)
print(np.linalg.inv(a))
# view_matrix_inverse = np.linalg.inv(view_matrix)
# print(view_matrix_inverse)
translation, quaternion = matrix_to_translation_quaternion(np.linalg.inv(a))
# 绕z轴旋转180度的四元数
rotation_z_180 = np.array([0, 0, 1, 0])

# 将两个四元数相乘
rotation = R.from_quat(quaternion) * R.from_quat(rotation_z_180)
new_quaternion = rotation.as_quat()
rotation = R.from_matrix(np.linalg.inv(a)[:3, :3])
euler_angles = rotation.as_euler('xyz', degrees=False)
print(euler_angles)
print("Translation:", translation)
print((new_quaternion))

def rotate_quaternion_around_z(quaternion, angle_degrees):
    """
    Rotate a quaternion around the z-axis by a given angle.
    
    Args:
        quaternion (np.ndarray): A 1x4 array of quaternion (x, y, z, w).
        angle_degrees (float): The angle to rotate around the z-axis in degrees.
        
    Returns:
        new_quaternion (np.ndarray): The resulting quaternion after rotation.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # Create the rotation quaternion for the z-axis rotation
    rotation_z = R.from_euler('z', angle_radians, degrees=False).as_quat()
    
    # Multiply the original quaternion by the z-axis rotation quaternion
    rotation = R.from_quat(quaternion) * R.from_quat(rotation_z)
    new_quaternion = rotation.as_quat()
    
    return new_quaternion
# 原始四元数
quaternion = np.array([0, -0.707105, 0.707108, 0])

# 绕z轴旋转90度
new_quaternion = rotate_quaternion_around_z(quaternion, 90)
print(new_quaternion)
# matrix = np.array([[ 0.31945667,  0.9471029 , -0.0307186 , -0.05195228],
# [ 0.20864381, -0.10192315, -0.9726662 ,  0.09869324],
# [-0.92434597,  0.30431554, -0.23016721,  0.5238725 ],
# [ 0.        ,  0.        ,  0.        ,  1.        ]])

# matrix = np.array([[ 0.31945667,  0.9471029 , -0.0307186 , -0.05195228],
# [ 0.20864381, -0.10192315, -0.9726662 ,  0.09869324],
# [-0.92434597,  0.30431554, -0.23016721,  0.5238725 ],
# [ 0.        ,  0.        ,  0.        ,  1.        ]])
# # matrix = np.array([[ 1,  0 , 0 , -0.05195228],
# # [ 0, 1, 0 ,  0.09869324],
# # [0,  0, 1,  0.5238725 ],
# # [ 0.        ,  0.        ,  0.        ,  1.        ]])
# translation, quaternion = matrix_to_translation_quaternion(matrix)
# print("Translation:", translation)
# print("Quaternion:", quaternion)
def quaternion_inverse(quaternion):
    """
    Calculate the inverse of a quaternion.
    
    Args:
        quaternion (np.ndarray): A 1x4 array of quaternion (x, y, z, w).
        
    Returns:
        inverse_quaternion (np.ndarray): The inverse of the quaternion.
    """
    # Calculate the conjugate of the quaternion
    conjugate = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
    
    # Calculate the norm (magnitude) of the quaternion
    norm = np.dot(quaternion, quaternion)
    
    # Calculate the inverse by dividing the conjugate by the norm
    inverse_quaternion = conjugate / norm
    
    return inverse_quaternion
def find_transform_quaternion(q1, q2):
    """
    Find the quaternion that transforms q1 to q2.
    
    Args:
        q1 (np.ndarray): The initial quaternion (x, y, z, w).
        q2 (np.ndarray): The target quaternion (x, y, z, w).
        
    Returns:
        transform_quaternion (np.ndarray): The quaternion that transforms q1 to q2.
    """
    # Calculate the inverse of q1
    q1_inverse = quaternion_inverse(q1)
    
    # Multiply q1_inverse by q2 to get the transform quaternion
    transform_quaternion = R.from_quat(q1_inverse) * R.from_quat(q2)
    
    return transform_quaternion.as_quat()
new = np.array([-0.00127045, -0.70704592,  0.70716083, -0.00282909])
new2 = np.array([0,1,0,0])
inverse_new = quaternion_inverse(quaternion)
print("Inverse Quaternion:", inverse_new)
transform_quaternion = find_transform_quaternion(new, new2)

print("Transform Quaternion:", transform_quaternion)