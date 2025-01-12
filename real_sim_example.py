import cv2
import numpy as np
from PIL import Image
from utils.dino_sam import get_segmentation
from utils.project_mesh import project_mesh_to_image
from pose_estimation.pose_estimation_client import get_pose
from simple_sim.unknow_obj.fit_unknow_obj import process_rgbd_with_mask

LABELS = ["can"]
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
            cv2.imwrite(f"{label}_{i}.png", object_mask_uint8)
            object_pose = self.obj_pose_estimation(rgb_image, depth_image, object_mask_uint8, label)
            if object_pose is not None:
                scene_dict["poses"].append(object_pose)
            else:
                print("unknown object")
                #TODO: handle unknown object
                scene_dict["poses"].append(np.eye(4))
        return scene_dict
import open3d as o3d
def sample_points(points, num_samples):
    if len(points) > num_samples:
        indices = np.random.choice(len(points), num_samples, replace=False)
        sampled_points = points[indices]
    else:
        sampled_points = points
    return sampled_points

def icp_registration(source_points, target_points, threshold=0.02, init_transformation=np.eye(4)):
    # num_samples = min(len(source_points), len(target_points))
    # source_points = sample_points(source_points, num_samples)
    # target_points = sample_points(target_points, num_samples)
    # Convert numpy arrays to Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    
    # Perform ICP registration
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    transformation = icp_result.transformation
    information = {
        "fitness": icp_result.fitness,
        "inlier_rmse": icp_result.inlier_rmse,
        "correspondence_set": icp_result.correspondence_set
    }
    
    return transformation, information

def create_rotation_z_matrix(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    
    rotation_matrix_z = np.array([
        [cos_angle, -sin_angle, 0, 0],
        [sin_angle,  cos_angle, 0, 0],
        [0,         0,         1, 0],
        [0,         0,         0, 1]
    ])
    return rotation_matrix_z

def create_rotation_x_matrix(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    
    rotation_matrix_x = np.array([
        [1, 0,         0,        0],
        [0, cos_angle, -sin_angle, 0],
        [0, sin_angle, cos_angle,  0],
        [0, 0,         0,        1]
    ])
    return rotation_matrix_x

def create_rotation_y_matrix(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    
    rotation_matrix_y = np.array([
        [ cos_angle, 0, sin_angle, 0],
        [ 0,         1, 0,         0],
        [-sin_angle, 0, cos_angle, 0],
        [ 0,         0, 0,         1]
    ])
    
    return rotation_matrix_y
from scipy.spatial.transform import Rotation as R

def get_poses(matrix_T):
    rotation = R.from_matrix(matrix_T[:3, :3])
    quaternion = rotation.as_quat()[[3, 0, 1, 2]]
    translation = matrix_T[:3, 3]
    return np.concatenate([translation, quaternion])

if __name__ == "__main__":
    example = RealSimExample()
    rgb_image_path = "/home/haowen/hw_mine/Real_Sim_Real/data/can_pour_1/3/scene_rgb_image/scene_1.jpg"
    depth_image_path = "/home/haowen/hw_mine/Real_Sim_Real/data/can_pour_1/3/scene_depth_image/scene_1.npy"
    
    # rgb_image, depth_image = example.read_image(rgb_image_path, depth_image_path)
    # scene_dict = example.run(rgb_image, depth_image)
    # print(scene_dict["poses"])
    # test = [[[-0.04988173395395279, 0.997614860534668, -0.047710709273815155, 0.008451678790152073], [0.2614016830921173, -0.033064473420381546, -0.964663565158844, 0.05631372332572937], [-0.9639402627944946, -0.060590710490942, -0.2591288387775421, 0.4898885488510132], [0.0, 0.0, 0.0, 1.0]]]
    # test = [[[0.9239045977592468, 0.38215941190719604, 0.01882772520184517, 0.009301328100264072], [0.007458145264536142, -0.06718483567237854, 0.9977127909660339, 0.05004724860191345], [0.38255012035369873, -0.9216508865356445, -0.06492260098457336, 0.5644583702087402], [0.0, 0.0, 0.0, 1.0]]]
    # test = [[[-0.5982769131660461, -0.7884963750839233, 0.14261187613010406, 0.017116937786340714], [0.7657954692840576, -0.6150311231613159, -0.18786659836769104, 0.043398667126894], [0.23584288358688354, -0.0031847022473812103, 0.9717860817909241, 0.5922662019729614], [0.0, 0.0, 0.0, 1.0]]]
    # [[[0.43723738193511963, 0.8989970684051514, -0.02505527436733246, 0.09150402992963791], [0.29450204968452454, -0.16944658756256104, -0.9405087232589722, 0.18733008205890656], [-0.8497599959373474, 0.4038466811180115, -0.3388448655605316, 0.6819711923599243], [0.0, 0.0, 0.0, 1.0]]]
    test = [[[-0.29022616147994995, 0.9559233784675598, -0.04448934271931648, -0.21637749671936035], [-0.5486288070678711, -0.12811657786369324, 0.8261916637420654, 0.23622964322566986], [0.7840760350227356, 0.26419055461883545, 0.5616299510002136, 0.5847076177597046], [0.0, 0.0, 0.0, 1.0]]]
    for pose in test:
        obj_path = "/home/haowen/hw_mine/Real_Sim_Real/simple_sim/asset/know/meshes/can/textured.obj"
        mesh = o3d.io.read_triangle_mesh(obj_path)
        pcd = mesh.sample_points_uniformly(number_of_points=25000)
        points = np.asarray(pcd.points)
        pose = np.array(pose)
        # for i, label in enumerate(scene_dict["labels"]):
        #     if label == "red kettle":
        #         object_mask = scene_dict["masks"][i]
        #         object_mask_uint8 = (object_mask).astype(np.uint8)
        #         now_pcd = process_rgbd_with_mask(np.load(depth_image_path), object_mask_uint8, example.camera_intrinsics)
        #         new_pose = icp_registration(points, np.asarray(now_pcd.points), pose)
        mask = cv2.imread("/home/haowen/hw_mine/Real_Sim_Real/can._0.png", cv2.IMREAD_GRAYSCALE)
        depth_now = np.load(depth_image_path)
    #     # filter_depth = np.where((depth_now < 0.5), depth_now, 0)
        now_pcd = process_rgbd_with_mask(depth_now, mask, example.camera_intrinsics)
        # o3d.visualization.draw_geometries([now_pcd])
    #     delta_y_180 = np.array([
    #     [-1,  0,  0,  0],
    #     [ 0,  1,  0,  0]scene_dict["poses"]
    #     [ 0,  0, -1,  0],
    #     [ 0,  0,  0,  1]
    # ])
    #     transformation_matrix_z_neg_30 = create_rotation_z_matrix(-50)
        delta_add = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0.03],
                [0, 0, 1, 0.02],
                [0, 0, 0, 1]
            ])
    #     pose = np.dot(np.array(pose),delta_y_180)
    #     pose = np.dot(pose, transformation_matrix_z_neg_30)
        pose = np.dot(pose, delta_add)
        delta_x_5 = create_rotation_x_matrix(5)
        pose = np.dot(pose, delta_x_5)
    #     delta_add_2 = np.array([
    #             [1, 0, 0, 0.01],
    #             [0, 1, 0, 0],
    #             [0, 0, 1, 0],
    #             [0, 0, 0, 1]
    #         ])
    #     pose = np.dot(pose, delta_add_2)
    #     delta_y = create_rotation_y_matrix(-10)
    #     pose = np.dot(pose, delta_y)
        new_pose, info = icp_registration(points, np.asarray(now_pcd.points),0.03, pose)
        
        projected_img, points_2d, depths = project_mesh_to_image(obj_path,
                                                                new_pose, example.camera_intrinsics,
                                                                cv2.imread(rgb_image_path))
        
        pose_new = new_pose.copy()
        print(get_poses(pose_new))
        import matplotlib.pyplot as plt
        # plt.imshow(np.load(depth_image_path))
        plt.imshow(cv2.cvtColor(projected_img, cv2.COLOR_BGR2RGB))
        plt.title("Projected 3D points")
        plt.show()