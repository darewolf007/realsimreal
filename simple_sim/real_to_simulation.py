import os
import time
import cv2
import pickle
import numpy as np
import sys
sys.path.insert(0, os.getcwd())
import robosuite.utils.camera_utils as CU
from simple_sim.base_env import SimpleEnv
from simple_sim.robotic_ik import mink_ik
from scipy.spatial.transform import Rotation as R
from simple_sim.sim_utils import transform_to_camera_frame, matrix_to_translation_quaternion, get_handeye_T
from simple_sim.sim_utils import quaternion_to_matrixT, adjust_orientation_to_z_up, crop_image, get_7Dof_pose, count_files_in_directory
from simple_sim.motion_planning import MotionPlanning
from simple_sim.camera_util import get_real_depth_map

class RealInSimulation:
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        self.env_info = env_info
        self.has_renderer = has_renderer
        self.last_action = None
        self.last_observation = None
        self.sub_task_static_point = []
        self.sub_task_moving_obj = []
        self.sub_task_idx = 0
        self.sub_task_step_num = 0
        self.sub_task_min_num = 25
        self.sub_task_max_num = 60
        self.all_task_step_num = 0 
        self.all_task_max_num = 65
        self.init_scene_xmlobj_pose()
        self.init_scene_camera_pose()
        self.init_motion_planning()
        self.env = SimpleEnv(robot, env_info, has_renderer = has_renderer, *args, **kwargs)
        self.init_invese_kinematics()

    def init_subtask_info(self):
        for subtask_obj in self.env_info['subtask_object_info']:
            static_point = self.env.sim.data.get_site_xpos(subtask_obj[1] + "_up_site").copy()
            self.sub_task_static_point.append(static_point)
            self.sub_task_moving_obj.append(subtask_obj[0])

    def init_motion_planning(self):
        self.planning = MotionPlanning()

    def init_invese_kinematics(self):
        mujoco_model = self.env.get_mujoco_model()
        self.mink_ik = mink_ik(mujoco_model)

    def init_scene_camera_pose(self):
        self.env_info['camera_info'] = {}
        self.env_info['camera_info']['sceneview'] = {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}
        if self.env_info['base_choose'] == "camera":
            for view_name, view_info in self.env_info['camera_info'].items():
                xyzw_quaternion = np.array(view_info["quat"])
                self.env_info['camera_info'][view_name]["quat"] = xyzw_quaternion[[3, 0, 1, 2]]
        elif self.env_info['base_choose'] == "robot":
            for view_name, view_info in self.env_info['camera_info'].items():
                camera_matrixT = quaternion_to_matrixT(view_info["quat"], view_info["pos"])
                robot_to_camera = np.dot(self.env_info['hand_eye'], camera_matrixT)
                translation, quaternion = matrix_to_translation_quaternion(robot_to_camera)
                quaternion = quaternion[[3, 0, 1, 2]]
                self.env_info['camera_info'][view_name]["pos"] = translation
                self.env_info['camera_info'][view_name]["quat"] = quaternion
        else:
            raise NotImplementedError

    def init_scene_xmlobj_pose(self):
        self.env_info['hand_eye'] = get_handeye_T(self.env_info['hand_eye_path'])
        for i in range(len(self.env_info['obj_info']['poses'])):
            pose_matrix = self.env_info['obj_info']['poses'][i]
            if pose_matrix.shape == (4, 4):
                self.env_info['obj_info']['poses'][i] = get_7Dof_pose(pose_matrix)
            elif pose_matrix.shape == (7,):
                continue
            else:
                raise NotImplementedError
        if self.env_info['base_choose'] == "camera":
            translation, quaternion = matrix_to_translation_quaternion(np.linalg.inv(self.env_info['hand_eye']))
            rotation_z_180 = np.array([0, 0, 1, 0])
            rotation = R.from_quat(quaternion) * R.from_quat(rotation_z_180)
            new_quaternion = rotation.as_quat()[[3, 0, 1, 2]]
            self.env_info['hand_eye'] = np.concatenate([translation, new_quaternion])
            self.env_info['robot_base_pose'] = self.env_info['hand_eye']
            for pose in self.env_info['obj_info']['poses']:
                obj_quat = pose[3:][[3, 0, 1, 2]]
                pose[3:] = obj_quat
        elif self.env_info['base_choose'] == "robot":
            self.env_info['robot_base_pose'] = np.array([0, 0, 0.025, 0, 0, 0, 1])
            if self.env_info['obj_pose_base'] == "camera":
                robot_to_camera = self.env_info['hand_eye']
                for pose in self.env_info['obj_info']['poses']:
                    camera_to_obj = quaternion_to_matrixT(pose[3:], pose[:3])
                    robot_to_obj = np.dot(robot_to_camera, camera_to_obj)
                    adjusted_rotation_matrix = adjust_orientation_to_z_up(robot_to_obj)
                    obj_translation, obj_quaternion = matrix_to_translation_quaternion(adjusted_rotation_matrix)
                    obj_quaternion = obj_quaternion[[3, 0, 1, 2]]
                    pose[:3] = obj_translation
                    pose[3:] = obj_quaternion
        else:
            raise NotImplementedError
        
    def pre_process_action(self, action, use_delta = True, use_joint_controller=False):
        if use_joint_controller:
            if use_delta:
                raise NotImplementedError
            else:
                self.last_action = action
                return action
        else:
            if use_delta:
                if self.last_action is None:
                    current_end_xpos = self.env.sim.data.get_site_xpos('robot0_attachment_site').copy()
                    current_end_xquat = self.env.sim.data.get_site_xmat('robot0_attachment_site').copy()
                    current_end_xquat = R.from_matrix(current_end_xquat).as_quat()
                    current_end_xquat = current_end_xquat[[3, 0, 1, 2]]
                    current_end_xpos[2] += 0.007
                    self.last_action = np.concatenate([current_end_xpos, current_end_xquat, np.array([-1])])
                delta_end_xpos = action[:3]
                delta_end_quat = action[3:7]
                end_xpos = self.last_action[:3] + delta_end_xpos
                end_quat = self.last_action[3:7] + delta_end_quat
                action = np.concatenate([end_xpos, end_quat, np.array([action[-1]])])
                self.last_action = action
            return action

    def pre_process_obs_image(self, observations, target_obj, step, is_collect = False, is_crop = True):
        new_observations = {}
        for camera_name in self.env_info['camera_names']:
            image = observations[camera_name + "_image"][..., ::-1]
            image = np.flip(image, axis=0)
            new_observations[camera_name + "_image"] = image
            if is_collect:
                cv2.imwrite(self.task_data_path + '/rgb_' + camera_name + "/" + str(step) + ".png", image)
        if self.env_info['camera_depths']:
            sceneview_depth_image = observations['sceneview_depth'][..., ::-1]
            sceneview_depth_image = np.flip(sceneview_depth_image, axis=0)
            depth_map = get_real_depth_map(sim=self.env.sim, depth_map=sceneview_depth_image)
            processed_depth_map = np.clip(depth_map, 0.0, 3.0)
            new_observations['sceneview_depth'] = processed_depth_map
            if is_collect:
                cv2.imwrite(self.task_data_path + '/depth_' + "sceneview" + "/" + str(step) + ".png", processed_depth_map)
        if is_crop:
            site_name = target_obj + "_center_site"
            obj_pos = self.env.sim.data.get_site_xpos(site_name).copy()
            world_to_camera = CU.get_camera_transform_matrix(
                sim=self.env.sim,
                camera_name="sceneview",
                camera_height=self.env_info['camera_heights'][0],
                camera_width=self.env_info['camera_widths'][0],
                )
            obj_pixel = CU.project_points_from_world_to_camera(
                points=obj_pos,
                world_to_camera_transform=world_to_camera,
                camera_height=self.env_info['camera_heights'][0],
                camera_width=self.env_info['camera_widths'][0],
                )
            crop_sceneview_image = crop_image(new_observations["sceneview_image"], (int(obj_pixel[1]), int(obj_pixel[0])), self.env_info['crop_image_size'])
            new_observations["crop_sceneview_image"] = crop_sceneview_image
            if is_collect:
                cv2.imwrite(self.task_data_path + '/crop_' + "sceneview" + "/" + str(step) + ".png", crop_sceneview_image)
        return new_observations
        # print("now", self.env.sim.data.get_site_xpos('robot0_attachment_site'))
        # print("gripperend", self.env.sim.data.get_site_xpos('gripper0_right_grip_site'))
        # print("gripper1", self.env.sim.data.get_site_xpos('gripper0_right_ft_frame'))
        # print("gripper2", self.env.sim.data.get_site_xpos('gripper0_right_grip_site_cylinder'))
        # print("can", self.env.sim.data.get_site_xpos('can_up_site'))

    def _step(self, action, use_joint_controller=False):
        if use_joint_controller:
            observations, reward, done, info = self.env.step(action)
        else:
            translation = action[:3]
            quaternion = action[3:7]
            qpos = self.mink_ik.ik(self.env.sim.data, translation, quaternion, 1/self.env_info['control_freq'])[:self.env.robots[0].dof]
            gripper_data = action[-1]
            action = np.concatenate([qpos, np.array([gripper_data])])
            observations, reward, done, info = self.env.step(action)
        return observations, reward, done, info

    def multi_step(self, action, use_delta=True, use_joint_controller=False, step_num=3, is_collect = False):
        step_action = self.pre_process_action(action, use_delta=use_delta, use_joint_controller=use_joint_controller)
        for i in range(step_num):
            observations, reward, done, info = self._step(step_action, use_joint_controller)
        self.all_task_step_num += 1
        subtask_id = self.get_subtask_id()
        info['subtask_id'] = subtask_id
        target_obj = self.env_info['subtask_object_info'][self.sub_task_idx][1]
        observations = self.pre_process_obs_image(observations, target_obj, self.all_task_step_num, is_collect = is_collect, is_crop = self.env_info['is_crop'])
        self.last_observation = observations
        if False:
            current_end_xpos = self.env.sim.data.get_site_xpos('robot0_attachment_site').copy()
            error = current_end_xpos - step_action[:3]
            end_quat = self.env.sim.data.get_site_xmat('robot0_attachment_site').copy()
            end_quat = R.from_matrix(end_quat).as_quat()
            end_quat = end_quat[[3, 0, 1, 2]]
            quat_error = end_quat - step_action[3:7]
            print("xpos", error)
            print("qpos", quat_error)
        return observations, reward, done, info

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        image = self.env.sim.render(camera_name=self.env_info['camera_names'][0], height=kwargs["height"], width=kwargs["height"])[..., ::-1]
        image = np.flip(image, axis=0)
        return image

    def get_subtask_id(self):
        subtask_id = self.get_subtask()
        if subtask_id != -1:
            self.sub_task_step_num += 1
        return subtask_id

    def replay_demonstration(self, use_joint_controller=False, is_collect = False, begin_step = 1):
        end_effector_action= []
        file_num = int(count_files_in_directory(self.env_info['data_path']) / 3)
        # self.env_info['robot_init_qpos'] = np.load(self.env_info['data_path'] + "joint_" + str(begin_step) + ".npy")
        # self.max_reward = 1
        # use_joint_controller=False,
        if is_collect:
            replay_data_save_path = self.env_info['replay_data_save_path']
            count = sum(os.path.isdir(os.path.join(replay_data_save_path, name)) for name in os.listdir(replay_data_save_path))
            self.task_data_path = replay_data_save_path + self.env_info['task_name'] + str(count +1)
            os.mkdir(self.task_data_path)
            for camera_name in self.env_info['camera_names']:
                os.mkdir(self.task_data_path + '/rgb_' + camera_name)
                if self.env_info['camera_depths'] and camera_name == "sceneview":
                    os.mkdir(self.task_data_path + '/depth_' + camera_name)
                    os.mkdir(self.task_data_path + '/crop_' + camera_name)
            self.task_step_data_path = self.task_data_path + '/data/'
            os.mkdir(self.task_step_data_path)

        for play_time in range(self.max_reward):
            print("update joint simulation xml obj pose")
            end_effector_action= []
            self.reset(update_env_info = True)
            for i in range(begin_step, file_num + 1):
                collect_data = np.load(self.env_info['data_path'] + "traj_" + str(i) + ".npy")
                joint_data = np.load(self.env_info['data_path'] + "joint_" + str(i) + ".npy")
                joint_data[0], joint_data[2] = joint_data[2], joint_data[0]
                if joint_data[0] == 0 and joint_data[1] == 0:
                    continue
                translation_data = collect_data[:3] + np.array([0.0, 0.0, 0.2])
                rotation_data = collect_data[3:-1]
                if collect_data[-1] == 0:
                    gripper_data = -1
                else:
                    gripper_data = 1
                if self.env_info['base_choose'] == "camera":
                    translation, xyzw_quaternion = transform_to_camera_frame(handeye_T, rotation_data, translation_data)
                    quaternion = xyzw_quaternion[[3, 0, 1, 2]]
                else:
                    translation = translation_data
                    quaternion = rotation_data[[3, 0, 1, 2]]
                if use_joint_controller:
                    action = np.concatenate([joint_data, np.array([gripper_data])])
                else:
                    action = np.concatenate([translation, quaternion, np.array([gripper_data])])
                if (self.last_action is not None and self.last_action[-1] != action[-1]) or i == file_num:
                    self.refine_obj_pose()
                    if play_time == 0:
                        break
                current_end_xpos = self.env.sim.data.get_site_xpos('robot0_attachment_site').copy()
                current_end_xquat = self.env.sim.data.get_site_xmat('robot0_attachment_site').copy()
                current_end_xquat = R.from_matrix(current_end_xquat).as_quat()
                current_end_xquat = current_end_xquat[[3, 0, 1, 2]]
                observations, reward, done, info = self.multi_step(action, use_delta=False, use_joint_controller=use_joint_controller, step_num=2, is_collect=False)
                end_xpos = self.env.sim.data.get_site_xpos('robot0_attachment_site').copy()
                end_quat = self.env.sim.data.get_site_xmat('robot0_attachment_site').copy()
                end_quat = R.from_matrix(end_quat).as_quat()
                end_quat = end_quat[[3, 0, 1, 2]]
                delta_end_xpos = end_xpos - current_end_xpos
                delta_end_quat = end_quat - current_end_xquat
                end_effector_action.append(np.concatenate([delta_end_xpos, delta_end_quat,np.array([gripper_data])]))
                if self.has_renderer:
                    self.env.render()
                print("self.sub_task_step_num", self.sub_task_step_num)
                print("sub id", self.sub_task_idx)
                if self.all_task_step_num == self.sub_task_max_num:
                    print(self.all_task_step_num)
                    break
                # break
        if use_joint_controller:
            print("update endeffector simulation xml obj pose")
            for play_time in range(self.max_reward):
                now_observation = self.reset(update_env_info = True)
                for step in range(len(end_effector_action)):
                    step_data = {}
                    action = end_effector_action[step]
                    if (self.last_action is not None and self.last_action[-1] != action[-1]):
                        self.refine_obj_pose()
                        if play_time == 0:
                            break
                    next_observations, reward, done, info = self.multi_step(action, use_delta=True, use_joint_controller=False, step_num=3)
                    if self.has_renderer:
                        self.env.render()
        np.save(self.env_info['replay_data_save_path'] + self.env_info['task_name'] +"obj_pose.npy", self.env_info['obj_info']['poses'])
        if use_joint_controller:
            print("collecting simulation data")
            now_observation = self.reset(update_env_info = True)
            now_qpos = self.env.sim.data.qpos[:self.env.robots[0].dof].copy()
            for step in range(len(end_effector_action)):
                step_data = {}
                action = end_effector_action[step]
                if (self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
                    step_data["rewards"] = 100
                else:
                    step_data["rewards"] = 0
                next_observations, reward, done, info = self.multi_step(action, use_delta=True, use_joint_controller=False, step_num=3, is_collect = is_collect)
                next_qpos = self.env.sim.data.qpos[:self.env.robots[0].dof].copy()
                target_obj = self.env_info['subtask_object_info'][self.sub_task_idx][1]
                if self.env_info['is_crop']:
                    step_data["obses"] = now_observation["crop_sceneview_image"]
                    step_data["next_obses"] = next_observations["crop_sceneview_image"]
                else:
                    step_data["obses"] = now_observation["sceneview_image"]
                    step_data["next_obses"] = next_observations["sceneview_image"]
                step_data["actions"] = action
                step_data["now_qpos"] = now_qpos
                step_data["next_qpos"] = next_qpos
                step_data["subtask_id"] = info['subtask_id']
                now_qpos = next_qpos
                if step == len(end_effector_action)-1:
                    step_data["rewards"] = 100
                    step_data["not_dones"] = False
                else:
                    step_data["not_dones"] = True
                now_observation = next_observations
                if self.has_renderer:
                    self.env.render()
                if is_collect:
                    with open(self.task_step_data_path + str(step + 1) + ".pkl", 'wb') as file:
                        pickle.dump(step_data, file)
        self.env.close()
    
    def reset(self, update_env_info = False):
        self.env.reset()
        self.env.sim.model.opt.integrator = 2
        if update_env_info:
            self.env.update_env_info(self.env_info)
        self.last_action = None
        self.sub_task_idx = 0
        self.in_subtask = False
        self.out_subtask = False
        self.sub_task_step_num = 0
        self.all_task_step_num = 0
        self.max_reward = len(self.env_info['subtask_language_info'])
        if not self.env_info['use_gravity']:
            self.env.sim.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        self.init_subtask_info()
        init_pose = self.env_info['robot_init_qpos']
        action = np.concatenate([init_pose, np.array([-1])])
        self._step(action, True)
        observations, reward, done, info = self._step(action, True)
        target_obj = self.env_info['subtask_object_info'][self.sub_task_idx][1]
        new_observation = self.pre_process_obs_image(observations, target_obj, 0, is_collect = False, is_crop = True)
        self.last_observation = new_observation
        return new_observation

    def is_in_subtask(self, threshold=0.1):
        moving_obj = self.sub_task_moving_obj[self.sub_task_idx]
        static_point_pos = self.sub_task_static_point[self.sub_task_idx]
        if moving_obj == "gripper":
            object_site_pos = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
        else:
            site_name = moving_obj + "_up_site"
            object_site_pos = self.env.sim.data.get_site_xpos(site_name).copy()
        distance = np.linalg.norm(static_point_pos - object_site_pos)
        if distance < threshold:
            return True
        else:
            return False
        
    def is_out_subtask(self, threshold=0.1):
        moving_obj = self.sub_task_moving_obj[self.sub_task_idx]
        static_point_pos = self.sub_task_static_point[self.sub_task_idx]
        if moving_obj == "gripper":
            object_site_pos = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
        else:
            site_name = moving_obj + "_up_site"
            object_site_pos = self.env.sim.data.get_site_xpos(site_name).copy()
        distance = np.linalg.norm(static_point_pos - object_site_pos)
        print("distance", distance)
        if distance < threshold:
            return False
        else:
            return True

    def get_subtask(self):
        if not self.in_subtask and not self.out_subtask:
            self.in_subtask = self.is_in_subtask()
            if not self.in_subtask:
                return -1
        if self.in_subtask and not self.out_subtask and self.sub_task_step_num > self.sub_task_min_num:
            self.out_subtask = self.is_out_subtask()
        if self.in_subtask and self.out_subtask:
            self.in_subtask = False
            self.out_subtask = False
            self.sub_task_idx += 1
            if self.sub_task_idx >= self.max_reward:
                self.sub_task_idx = self.max_reward - 1
            self.sub_task_step_num = 0
        if not self.in_subtask and self.out_subtask:
            print("wrong")
            raise NotImplementedError
        return self.sub_task_idx
           
    def refine_obj_pose(self):
        print("refine pose")
        moving_obj = self.sub_task_moving_obj[self.sub_task_idx]
        if moving_obj == "gripper":
            object_site_pos = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
        else:
            site_name = moving_obj + "_up_site"
            object_site_pos = self.env.sim.data.get_site_xpos(site_name).copy()
        target_obj = self.env_info['subtask_object_info'][self.sub_task_idx][1]
        idx = self.env_info['obj_info']['labels'].index(target_obj)
        self.env_info['obj_info']['poses'][idx][:2] = object_site_pos[:2]
    
    def ask_target_pos(self, current_image, target_object, target_object_site_pos, random_sample=True):
        if random_sample == True:
            delta_z = np.random.uniform(0.15, 0.16)
            print("target_si", target_object_site_pos)
            target_position = target_object_site_pos + np.array([0, 0, delta_z])
            print("target", target_position)
            return target_position

    def is_planning(self, manipulation_object, target_object_site_pos, threshold=0.05):
        if manipulation_object == "gripper":
            manipulation_object_site_pos = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
        else:
            site_name = manipulation_object + "_up_site"
            manipulation_object_site_pos = self.env.sim.data.get_site_xpos(site_name).copy()
        distance = np.linalg.norm(manipulation_object_site_pos - target_object_site_pos)
        if distance < threshold:
            print("no need planning")
            return False
        else:
            return True
        
    def run_planning(self, manipulation_object, target_object, current_image):
        target_object_site_pos = self.env.sim.data.get_site_xpos(target_object + "_up_site").copy()
        print("now", self.env.sim.data.get_site_xpos('robot0_attachment_site'))
        print("gripperend", self.env.sim.data.get_site_xpos('gripper0_right_grip_site'))
        print("Upup", self.env.sim.data.get_site_xpos(target_object + "_up_up_site"))
        if self.is_planning(manipulation_object, target_object_site_pos):
            start_position = self.env.sim.data.get_site_xpos('robot0_attachment_site').copy()
            target_position = self.ask_target_pos(current_image, target_object, target_object_site_pos)
            xyz_trajectory = self.planning.plan(start_position, target_position)
            delata_action = np.diff(xyz_trajectory, axis=0, prepend=start_position.reshape(1, -1))[1:]
            print(xyz_trajectory)
            return delata_action
        else:
            return None

    @property
    def observation_shape(self):
        if self.env_info['crop_image_size'] is not None:
            return (3, self.env_info['crop_image_size'][0], self.env_info['crop_image_size'][1])
        else:
            return (3, self.env_info['camera_heights'][0], self.env_info['camera_widths'][0])
        
    @property
    def action_shape(self):
        if self.env_info['use_joint_controller']:
            return (self.env.robots[0].dof,)
        else:
            return (8,)
        
if __name__ == "__main__":
    task_name = "Pour can into a cup"
    subtask_1 = "Pick up the can"
    subtask_2 = "Pour the can into the cup"
    subtask_1_obj = ["gripper", "can"]
    subtask_2_obj = ["can", "cup"]
    base_path = os.path.dirname(os.path.realpath(__file__))
    handeye_T_path = os.path.join(base_path, "../configs/ur5_kinect_handeyecalibration_eye_on_base.yaml")
    handeye_T = get_handeye_T(handeye_T_path)
    robot_init_pose = np.array([ -1.30487138, -1.69159379, 1.7358554 , -1.55820926, -1.51700765,
       -0.55815155])
    can_pose = np.array([[-0.29022616147994995, 0.9859233784675598, -0.04448934271931648, -0.21637749671936035], [-0.5486288070678711, -0.12811657786369324, 0.8261916637420654, 0.23622964322566986], [0.7840760350227356, 0.26419055461883545, 0.5616299510002136, 0.5847076177597046], [0.0, 0.0, 0.0, 1.0]])
    # can_pose_quat = get_7Dof_pose(can_pose)
    # can_pose_quat = np.array([-0.22424821, 0.18342368,  0.59452748,  0.33435988,  0.48682123, 0.67549918, -0.44148546])
    # can_pose_quat = np.array([-0.0660707, 0.16830145,  0.52134751, -0.44148546,  0.33435988,  0.48682123, 0.67549918])
    cup_pose = np.array([[0.43723738193511963, 0.8989970684051514, -0.02505527436733246, 0.09150402992963791], [0.29450204968452454, -0.16944658756256104, -0.9405087232589722, 0.18733008205890656], [-0.8497599959373474, 0.4038466811180115, -0.3388448655605316, 0.6819711923599243], [0.0, 0.0, 0.0, 1.0]])
    # cup_pose_quat = get_7Dof_pose(cup_pose)
    scene_dict = {"labels": ["can", "cup"], "poses": [can_pose, cup_pose], "grasp_obj": [True, False]}
    replay_data_save_path = os.path.join(base_path, "../data/sim_data/")
    env_info = {}
    env_info['obj_pose_base'] = "camera"
    env_info['replay_data_save_path'] = replay_data_save_path
    env_info['task_name'] = task_name
    env_info['subtask_language_info'] = [subtask_1, subtask_2]
    env_info['subtask_object_info'] = [subtask_1_obj, subtask_2_obj]
    env_info['hand_eye_path'] = handeye_T_path
    env_info['hand_eye'] = handeye_T
    env_info['obj_info'] = scene_dict
    env_info['use_gravity'] = True
    env_info['data_path'] = "/home/haowen/hw_mine/Real_Sim_Real/data/real_data/pour_all/8/traj/"
    begin_step = 35
    # env_info['base_choose'] = "camera"
    env_info['base_choose'] = "robot"
    robot_init_pose = np.load("/home/haowen/hw_mine/Real_Sim_Real/data/real_data/pour_all/8/traj/" + "joint_34.npy")
    robot_init_pose[0], robot_init_pose[2] = robot_init_pose[2], robot_init_pose[0]
    env_info['robot_init_qpos'] = robot_init_pose
    env_info['max_reward'] = 1
    env_info['camera_depths'] = True
    env_info['is_crop'] = False
    env_info['crop_image_size'] = (768, 768)
    env_info['camera_heights'] = [768*2, 1536, 1536, 1536]
    env_info['camera_widths'] = [768*2, 2048, 2048, 2048]
    env_info['camera_names'] = ["sceneview", "birdview", "frontview", "rightview"]
    env_info['has_renderer'] = True
    env_info['control_freq'] = 20
    env_info['init_noise'] = False
    env_info['init_translation_noise_bounds'] = (-0.03, 0.03)
    env_info['init_rotation_noise_bounds'] = (-50, 50)
    test_real = RealInSimulation("UR5e",
                                 env_info,
                                 has_renderer=env_info['has_renderer'],
                                 has_offscreen_renderer=True,
                                 render_camera="sceneview",
                                 ignore_done=True,
                                 use_camera_obs=True,
                                 camera_depths=env_info['camera_depths'],
                                 control_freq=env_info['control_freq'],
                                 renderer="mjviewer",
                                 camera_heights=env_info['camera_heights'],
                                 camera_widths=env_info['camera_widths'],
                                 camera_names=env_info['camera_names'],)
    test_real.reset()
    while(True):
        test_real.reset()
        for _ in range(10):
            observations,_,_,_ = test_real.multi_step(np.array([0, 0, 0, 0, 0, 0, 1]))
        #     import cv2
        #     cv2.imshow("sceneview", observations['sceneview_image'])
        #     cv2.waitKey(2)
        # cv2.destroyAllWindows()
        # import matplotlib.pyplot as plt
        # plt.imshow(observations['crop_sceneview_image'])
        # plt.show()
    test_real.replay_demonstration(use_joint_controller= True, is_collect=True, begin_step=begin_step)