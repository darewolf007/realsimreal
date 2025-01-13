import os
import time
import cv2
import pickle
import numpy as np
from simple_sim.real_to_simulation import RealInSimulation
from reward_model.online_reward_model import ask_grasp_subtask, ask_pour_subtask
from agent_policy.few_shot_RL.policy import FewDemoPolicy

class PourSimulation(RealInSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)

    # def step(self, action, use_joint_controller=False):
    #     pass
    #     obs = pass
    #     reward = pass
    #     done = pass
    #     info = pass

    def is_done(self):
        pass

    def is_grasp(self):
        pass

    def is_pour(self):
        pass

    def train(self):
        pass

if __name__ == "__main__":
    task_name = "Pour can into a cup"
    subtask_1 = "Pick up the can"
    subtask_2 = "Pour the can into the cup"
    subtask_1_obj = ["gripper", "can"]
    subtask_2_obj = ["can", "cup"]
    base_path = os.path.dirname(os.path.realpath(__file__))
    handeye_T_path = os.path.join(base_path, "./configs/ur5_kinect_handeyecalibration_eye_on_base.yaml")
    # handeye_T = get_handeye_T(handeye_T_path)
    robot_init_pose = np.array([ -1.30487138, -1.69159379, 1.7358554 , -1.55820926, -1.51700765,
       -0.55815155])
    can_pose = np.array([[-0.29022616147994995, 0.9859233784675598, -0.04448934271931648, -0.21637749671936035], [-0.5486288070678711, -0.12811657786369324, 0.8261916637420654, 0.23622964322566986], [0.7840760350227356, 0.26419055461883545, 0.5616299510002136, 0.5847076177597046], [0.0, 0.0, 0.0, 1.0]])
    # can_pose_quat = get_7Dof_pose(can_pose)
    # can_pose_quat = np.array([-0.22424821, 0.18342368,  0.59452748,  0.33435988,  0.48682123, 0.67549918, -0.44148546])
    # can_pose_quat = np.array([-0.0660707, 0.16830145,  0.52134751, -0.44148546,  0.33435988,  0.48682123, 0.67549918])
    cup_pose = np.array([[0.43723738193511963, 0.8989970684051514, -0.02505527436733246, 0.09150402992963791], [0.29450204968452454, -0.16944658756256104, -0.9405087232589722, 0.18733008205890656], [-0.8497599959373474, 0.4038466811180115, -0.3388448655605316, 0.6819711923599243], [0.0, 0.0, 0.0, 1.0]])
    # cup_pose_quat = get_7Dof_pose(cup_pose)
    scene_dict = {"labels": ["can", "cup"], "poses": [can_pose, cup_pose], "grasp_obj": [True, False]}
    data_save_path = os.path.join(base_path, "../data/sim_data/")
    env_info = {}
    env_info['data_save_path'] = data_save_path
    env_info['task_name'] = task_name
    env_info['subtask_language_info'] = [subtask_1, subtask_2]
    env_info['subtask_object_info'] = [subtask_1_obj, subtask_2_obj]
    env_info['hand_eye_path'] = handeye_T_path
    env_info['obj_info'] = scene_dict
    env_info['use_gravity'] = True
    env_info['data_path'] = "/home/haowen/hw_mine/Real_Sim_Real/data/pour_all/8/traj/"
    # env_info['base_choose'] = "camera"
    env_info['base_choose'] = "robot"
    env_info['robot_init_qpos'] = robot_init_pose
    env_info['max_reward'] = 1
    env_info['camera_depths'] = True
    env_info['crop_image_size'] = (770, 770)
    env_info['camera_heights'] = [1536, 1536, 1536, 1536]
    env_info['camera_widths'] = [2048, 2048, 2048, 2048]
    env_info['camera_names'] = ["sceneview", "birdview", "frontview", "rightview"]
    env_info['has_renderer'] = True
    env_info['control_freq'] = 20
    test_real = PourSimulation("UR5e",
                                 env_info,
                                 has_renderer=env_info['has_renderer'],
                                 has_offscreen_renderer=True,
                                 render_camera=env_info['camera_names'][1],
                                 ignore_done=True,
                                 use_camera_obs=True,
                                 camera_depths=env_info['camera_depths'],
                                 control_freq=env_info['control_freq'],
                                 renderer="mjviewer",
                                 camera_heights=env_info['camera_heights'],
                                 camera_widths=env_info['camera_widths'],
                                 camera_names=env_info['camera_names'],)
    test_real.reset()
    # test_real.replay_demonstration(use_joint_controller= True, is_collect=True)
