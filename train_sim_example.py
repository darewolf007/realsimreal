import os
import time
import cv2
import pickle
import torch
import numpy as np
import gymnasium as gym
from utils.data_convert_pt import convert_pickles_to_pt
from utils.image_util import resize_image, save_image_pkl
from simple_sim.real_to_simulation import RealInSimulation
from reward_model.online_reward_model import ask_grasp_subtask, ask_pour_subtask
from agent_policy.few_shot_RL.policy import FewDemoPolicy

class PourSimulation(RealInSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.grasp_flag = False
        self.task_data_path = "/home/haowen/hw_mine/Real_Sim_Real/experiments/Pour_can_into_cup/promot_data/all"
        if env_info['use_joint_controller']:
            self.action_space = gym.spaces.Box(low=-env_info['max_action'], high=env_info['max_action'], shape=(self.env.robots[0].dof,), dtype=float)
        else:
            self.action_space = gym.spaces.Box(low=-env_info['max_action'], high=env_info['max_action'], shape=(8,), dtype=float)

    def step(self, action, use_delta=True, use_joint_controller=False):
        action[:7] = np.clip(action[:7], -self.env_info['max_action'], self.env_info['max_action'])
        action[-1] = 1 if action[-1] > 0 else -1
        reward = self.update_reward(action)
        next_observation, _, _, info = super().multi_step(action, use_delta, use_joint_controller, is_collect=True)
        # obs = next_observation['crop_sceneview_image']
        obs = resize_image(next_observation['crop_sceneview_image'], 1/6)
        obs = np.transpose(obs, (2, 0, 1))
        done = self.update_done(next_observation)
        if done:
            return obs, 100, True, info
        return obs, reward, False, info
    
    def update_reward(self, action):
        if not self.grasp_flag and (self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            print("in update reward")
            self.grasp_flag = self.is_grasp(self.last_observation)
            if self.grasp_flag:
                return 100
        return -1

    def update_done(self, next_observation):
        if self.grasp_flag and (self.all_task_max_num - self.all_task_step_num < 30):
            if self.all_task_step_num % 10 == 0:
                print("in update done")
                done = self.is_done(next_observation)
                return done
        return False

    def is_done(self, observations):
        moving_obj = self.env_info['subtask_object_info'][self.sub_task_idx][0]
        target_obj = self.env_info['subtask_object_info'][self.sub_task_idx][1]
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.5),
            "right_view": resize_image(observations["rightview_image"], 0.5),
            "bird_view": resize_image(observations["birdview_image"], 0.5)
        }
        done_flag = ask_pour_subtask(image_dict, moving_obj, target_obj)
        if self.env_info['save_online_image']:
            image_dict['result'] = done_flag
            save_image_pkl(image_dict, self.env_info['online_data_save_path'] + "/done/", True)
        return done_flag

    def is_grasp(self, observations):
        moving_obj = self.env_info['subtask_object_info'][self.sub_task_idx][0]
        target_obj = self.env_info['subtask_object_info'][self.sub_task_idx][1]
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.5),
            "right_view": resize_image(observations["rightview_image"], 0.5),
            "bird_view": resize_image(observations["birdview_image"], 0.5)
        }
        grasp_flag = ask_grasp_subtask(image_dict, moving_obj, target_obj)
        if self.env_info['save_online_image']:
            image_dict['result'] = grasp_flag
            save_image_pkl(image_dict, self.env_info['online_data_save_path'] + "/grasp/", True)
        return grasp_flag

    def is_pour(self):
        pass

    def reset(self):
        self.grasp_flag = False
        observation = super().reset()
        obs = resize_image(observation['crop_sceneview_image'], 1/6)
        obs = np.transpose(obs, (2, 0, 1))
        return obs

def set_params():
    task_name = "Pour_can_into_cup"
    subtask_1 = "Pick up the can"
    subtask_2 = "Pour the can into the cup"
    subtask_1_obj = ["gripper", "can"]
    subtask_2_obj = ["can", "cup"]
    base_path = os.path.dirname(os.path.realpath(__file__))
    work_dir = os.path.join(base_path, "./experiments/" + task_name)
    real_data_dir = os.path.join(base_path, "./data/sim_data/" + task_name)
    handeye_T_path = os.path.join(base_path, "./configs/ur5_kinect_handeyecalibration_eye_on_base.yaml")
    replay_data_save_path = os.path.join(base_path, "./data/sim_data/")
    robot_init_pose = np.array([-1.28626711, -1.91235318,  2.03487999, -1.58658661, -1.50936824, -0.32230741])
    env_info = {}
    # env_info['obj_pose_base'] = "camera"
    # can_pose = np.array([[-0.29022616147994995, 0.9859233784675598, -0.04448934271931648, -0.21637749671936035], [-0.5486288070678711, -0.12811657786369324, 0.8261916637420654, 0.23622964322566986], [0.7840760350227356, 0.26419055461883545, 0.5616299510002136, 0.5847076177597046], [0.0, 0.0, 0.0, 1.0]])
    # cup_pose = np.array([[0.43723738193511963, 0.8989970684051514, -0.02505527436733246, 0.09150402992963791], [0.29450204968452454, -0.16944658756256104, -0.9405087232589722, 0.18733008205890656], [-0.8497599959373474, 0.4038466811180115, -0.3388448655605316, 0.6819711923599243], [0.0, 0.0, 0.0, 1.0]])
    env_info['obj_pose_base'] = "robot"
    can_pose = np.array([-2.58006106e-01,  4.91104923e-01,  4.95361287e-02,  5.03430916e-02, 1.38954074e-15, -3.13515130e-16,  9.98731983e-01])
    cup_pose = np.array([-3.99759997e-01,  2.25097344e-01,  4.50455912e-02,  1.47272580e-01, 3.50769505e-18,  5.26154258e-18,  9.89095944e-01])
    scene_dict = {"labels": ["can", "cup"], "poses": [can_pose, cup_pose], "grasp_obj": [True, False]}
    env_info['online_data_save_path'] = os.path.join(work_dir, "promot_data")
    env_info['replay_data_save_path'] = replay_data_save_path
    env_info['task_name'] = task_name
    env_info['subtask_language_info'] = [subtask_1, subtask_2]
    env_info['subtask_object_info'] = [subtask_1_obj, subtask_2_obj]
    env_info['hand_eye_path'] = handeye_T_path
    env_info['obj_info'] = scene_dict
    env_info['use_gravity'] = True
    env_info['is_crop'] = True
    env_info['data_path'] = None
    env_info['save_online_image'] = True
    # env_info['base_choose'] = "camera"
    env_info['base_choose'] = "robot"
    env_info['robot_init_qpos'] = robot_init_pose
    env_info['max_reward'] = 1
    env_info['camera_depths'] = True
    env_info['crop_image_size'] = (768, 768)
    env_info['camera_heights'] = [1536, 1536, 1536, 1536]
    env_info['camera_widths'] = [2048, 2048, 2048, 2048]
    env_info['camera_names'] = ["sceneview", "birdview", "frontview", "rightview"]
    env_info['has_renderer'] = True
    env_info['control_freq'] = 20
    env_info['use_joint_controller'] = False
    env_info['max_action'] = 0.1
    env_info['init_noise'] = False
    env_info['init_translation_noise_bounds'] = (-0.03, 0.003)
    env_info['init_rotation_noise_bounds'] = (-5, 5)
    

    policy_params = {
    "work_dir": work_dir,
    "task_name": task_name,
    "sub_task_promot": [subtask_1, subtask_2],
    "replay_buffer_capacity": 100000,
    "replay_buffer_load_dir": replay_data_save_path + "pt_data",
    "replay_buffer_keep_loaded": True,
    "pretrain_mode": None,
    "pre_transform_image_size": 128,
    "image_size": 112,
    "cameras": [0],
    "observation_type": "pixel",
    "reward_type": "dense",
    "control": None,
    "action_repeat": 1,
    "frame_stack": 1,
    "num_updates": 1,
    "model_dir": None,
    "model_step": 40000,
    "agent_name": "dino_e2c_sac",
    "init_steps": 0,
    "num_train_steps": 10000,
    "batch_size": 128,
    "hidden_dim": 1024,
    "eval_freq": 1000,
    "num_eval_episodes": 2,
    "critic_lr": 1e-3,
    "critic_beta": 0.9,
    "critic_tau": 0.01,
    "critic_target_update_freq": 2,
    "actor_lr": 1e-3,
    "actor_beta": 0.9,
    "actor_log_std_min": -10,
    "actor_log_std_max": 2,
    "actor_update_freq": 2,
    "encoder_type": "pixel",
    "encoder_feature_dim": 32,
    "encoder_tau": 0.05,
    "num_layers": 4,
    "num_filters": 32,
    "latent_dim": 128,
    "discount": 0.99,
    "init_temperature": 0.1,
    "alpha_lr": 1e-4,
    "alpha_beta": 0.5,
    "seed": 1,
    "save_tb": True,
    "save_buffer": True,
    "save_video": False,
    "save_sac": True,
    "detach_encoder": False,
    "v_clip_low": None,
    "v_clip_high": None,
    "action_noise": None,
    "final_demo_density": None,
    "data_augs": "crop",
    "log_interval": 200,
    "conv_layer_norm": True,
    "p_reward": 1,
    }
    return env_info, policy_params

def trainer():
    env_info, policy_params = set_params()
    env = PourSimulation("UR5e",
                        env_info,
                        has_renderer=env_info['has_renderer'],
                        has_offscreen_renderer=True,
                        render_camera=env_info['camera_names'][0],
                        ignore_done=True,
                        use_camera_obs=True,
                        camera_depths=env_info['camera_depths'],
                        control_freq=env_info['control_freq'],
                        renderer="mjviewer",
                        camera_heights=env_info['camera_heights'],
                        camera_widths=env_info['camera_widths'],
                        camera_names=env_info['camera_names'],)
    policy = FewDemoPolicy(env, torch.device("cuda"), policy_params)
    policy.train()

if __name__ == "__main__":
    # convert_pickles_to_pt()
    trainer()

    