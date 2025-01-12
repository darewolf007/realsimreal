import cv2
import numpy as np
import open3d as o3d

def project_mesh_to_image(obj_path, pose, intrinsics, rgb_image, depth_scale=1000.0):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    points = np.asarray(pcd.points)
    R = pose[:3, :3]
    t = pose[:3, 3]
    transformed_points = (R @ points.T).T + t
    now_pcd = o3d.geometry.PointCloud()
    now_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    # o3d.visualization.draw_geometries([now_pcd])
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    proj_points = []
    depths = []
    for pt3d in transformed_points:
        z = pt3d[2]
        if z <= 0:
            continue   
        x = int((fx * pt3d[0] / z) + cx)
        y = int((fy * pt3d[1] / z) + cy)
        if 0 <= x < rgb_image.shape[1] and 0 <= y < rgb_image.shape[0]:
            proj_points.append([x, y])
            depths.append(z)
    proj_points = np.array(proj_points)
    depths = np.array(depths)
    vis_img = rgb_image.copy()
    depth_normalized = (depths - depths.min()) / (depths.max() - depths.min())
    for (x, y), d in zip(proj_points, depth_normalized):
        color = (0, int(255 * (1-d)), int(255 * d))
        cv2.circle(vis_img, (x, y), 1, color, -1)
    return vis_img, proj_points, depths