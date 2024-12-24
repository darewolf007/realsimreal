# import numpy as np
# def calculate_fovy(intrinsic_matrix, image_height):
#     """
#     根据相机内参矩阵和图像高度计算 fovy 参数。

#     :param intrinsic_matrix: 相机内参矩阵 (3x3)
#     :param image_height: 图像垂直分辨率
#     :return: fovy 参数，单位为度数
#     """
#     fy = intrinsic_matrix[1, 1]  # 内参矩阵中 f_y
#     fovy = 2 * np.arctan(image_height / (2 * fy)) * (180 / np.pi)
#     a = np.degrees(2 * np.arctan(image_height / (2 * fy)))
#     return a
# intrinsics = np.array([[978.7357, 0.0, 1030.9428], [0.0, 979.040, 766.455], [0.0, 0.0, 1.0]])

# print(calculate_fovy(intrinsics, 1536))
# fx = 978.7357
# fy = 979.040
# delta_cx = 2048/2 - 1030.9428
# cy = 1536/2 - 766.455
# delta_x = (delta_cx)/fx
# delta_y = (cy)/fy
# theta_x = np.arctan(delta_x)
# theta_y = np.arctan(delta_y)
# from scipy.spatial.transform import Rotation as R
# rotation = R.from_euler('yx', [theta_y, theta_x], degrees=False)
# quaternion = rotation.as_quat()
# print(quaternion)
import pybullet as p
import pybullet_data
import numpy as np

def setup_scene_with_urdf():
    # 初始化PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 设置物理属性
    # p.setGravity(0, 0, -9.8)

    # 加载地面
    p.loadURDF("plane.urdf")

    # 加载URDF文件
    urdf_path = "/home/haowen/hw_mine/Real_Sim_Real/data/pybullet/urdf/test.urdf"
    p.loadURDF(urdf_path, basePosition=[0, 0, 0], baseOrientation= [0, 0, 0, 1])

    # 设置相机内参
    fx = 978.7357  # 焦距 (像素)
    fy = 979.040  # 焦距 (像素)
    cx = 1030.9428  # 主点x (像素)
    cy = 766.455  # 主点y (像素)
    width = 2048
    height = 1536

    # 内参矩阵转换为透视矩阵
    near = 0.1
    far = 100
    projection_maix = [
        2 * fx / width, 0, 0, 0,
        0, 2 * fy / height, 0, 0,
        1 - 2 * cx / width, 2 * cy / height - 1, -(far + near) / (far - near), -1,
        0, 0, -2 * far * near / (far - near), 0
    ]

    # 设置视图矩阵（单位矩阵，因为世界坐标系即相机坐标系）
    view_matrix = [
        1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1
    ]

    # 渲染相机视图
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )
    camera_position = [0, 0, 1]
    camera_target = [0, 0, 0]
    camera_up = [0, 1, 0]

    # 绘制相机位置和坐标系
    # p.addUserDebugLine(camera_position, camera_target, [1, 0, 0], 2)  # X轴 红色
    # p.addUserDebugLine(camera_position, [camera_position[0], camera_position[1] + 1, camera_position[2]], [0, 1, 0], 2)  # Y轴 绿色
    # p.addUserDebugLine(camera_position, [camera_position[0], camera_position[1], camera_position[2] + 1], [0, 0, 1], 2)  # Z轴 蓝色

    print("Camera setup complete. Press Q to quit.")
    while True:
        keys = p.getKeyboardEvents()
        if ord('q') in keys:
            break

    p.disconnect()

if __name__ == "__main__":
    setup_scene_with_urdf()