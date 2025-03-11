import os
import time
import cv2
import pickle
import torch
import argparse
import numpy as np
from utils.image_util import resize_image, save_image_pkl
from simple_sim.real_to_simulation import RealInSimulation

class PickSimulation(RealInSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.gripper_change_num = 0   

    def step(self, action, use_joint_controller=False):
        action[:3] = np.clip(action[:3], -self.env_info['max_action'], self.env_info['max_action'])
        action[:3] = action[:3]/100
        action[-1] = 1 if action[-1] > 0 else -1
        observation, _, _, info = super().multi_step(action, self.env_info['use_delta'], use_joint_controller, is_collect=False, step_num=1, use_euler= self.env_info['use_euler'])
        obs = observation['sceneview_image']
        done = self.is_sucess(info, action)
        reward = self.reward(info, action)
        if done:
            info['is_success'] = True
            return obs, reward, done, info
        else:
            info['is_success'] = False
        return obs, reward, done, info 

    def reward(self, info, action):
        raise NotImplementedError

    def is_sucess(self, info, action):
        raise NotImplementedError

    def reset(self):
        observation = super().reset()
        obs = observation['sceneview_image']
        self.gripper_change_num = 0
        return obs
    
class PickBananaSimulation(PickSimulation):
    def reward(self, info, action):
        if self.env_info['reward_type'] == "sparse":
            reaching_reward = -1
        else:
            reaching_reward = (- info["gripper_banana"]) * 10
        grasp_reward = 0
        if (self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            if info["gripper_banana"] > 0.07:
                self.gripper_change_num += 1
        if self.is_sucess(info, action):
            grasp_reward = 100 - self.gripper_change_num * 10
        return reaching_reward + grasp_reward

    def is_sucess(self, info, action):
        if action[-1] == 1 and (
            0.1 > info["delta_gripper"] and info["delta_gripper"] > 0.04) and (
                info["gripper_banana"] < 0.05
            ):
            return True
        else:
            return False
    