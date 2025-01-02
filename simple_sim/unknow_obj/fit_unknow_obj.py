import cv2
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def depth_to_pointcloud(depth_img, intrinsic, depth_scale=1.0, max_depth=3.0):
    depth_img = depth_img.astype(np.float32) * depth_scale
    rows, cols = depth_img.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=False)
    z = depth_img
    x = (c - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (r - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.dstack((x, y, z)).reshape(-1, 3)
    valid_mask = (points[:, 2] > 0) & (points[:, 2] <= max_depth)
    points = points[valid_mask]
    return points

def reproject_geometry_to_image(vertices, intrinsic, rgb_image):
    rgb_image = np.array(rgb_image)
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    projected_points = []
    for vertex in vertices:
        x, y, z = vertex
        if z > 0:
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)
            if 0 <= u < rgb_image.shape[1] and 0 <= v < rgb_image.shape[0]:
                projected_points.append((u, v))
    projected_img = rgb_image.copy()
    for u, v in projected_points:
        cv2.circle(projected_img, (u, v), radius=5, color=(0, 255, 0), thickness=-1)
    return projected_img

def process_rgbd_with_mask(depth_img, mask_image, intrinsic, other_mask=[], filter=True, depth_scale = 1.0):
    masked_depth = depth_img.copy()
    masked_depth[mask_image == 0] = 0
    if other_mask:
        for om in other_mask:
            masked_depth[om > 0] = 0
    point_cloud = depth_to_pointcloud(masked_depth, intrinsic, depth_scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    if filter:
        points = np.asarray(pcd.points)
        db = DBSCAN(
            eps=0.02,  # 2cm radius
            min_samples=10,  # Minimum points in cluster
            algorithm='ball_tree',  # More efficient for 3D data
        ).fit(points)
        labels = db.labels_
        max_label = labels.max()
        if max_label >= 0:
            largest_cluster_label = np.bincount(labels[labels >= 0]).argmax()
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            mask = labels == largest_cluster_label
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            return filtered_pcd
    return pcd


def fit_geometry_with_ransac(pcd, geometry_type="plane"):
    if geometry_type == "plane":
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )
        plane_cloud = pcd.select_by_index(inliers)
        plane_mesh, _ = plane_cloud.compute_convex_hull()
        plane_mesh.orient_triangles()
        return plane_mesh

    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

def project_obj_to_image():
    pass

def process_unknow_obj(rgb_image, depth_img, mask_image, intrinsic, label, other_mask=[], filter=True, depth_scale = 1.0, obj_type="plane", render=False):
    pcd = process_rgbd_with_mask(depth_img, mask_image, intrinsic, other_mask, filter, depth_scale)
    obj_mesh = fit_geometry_with_ransac(pcd, obj_type)
    if render:
        o3d.visualization.draw_geometries([obj_mesh])
        vertices = np.asarray(obj_mesh.sample_points_uniformly(number_of_points=5000).points)
        projected_img = reproject_geometry_to_image(vertices, intrinsic, rgb_image)
        plt.imshow(cv2.cvtColor(projected_img, cv2.COLOR_BGR2RGB))
        plt.show()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    mesh_dir = f'{code_dir}/../asset/unknow/meshes/{label}.obj'
    o3d.io.write_triangle_mesh(mesh_dir, obj_mesh)
    return obj_mesh

