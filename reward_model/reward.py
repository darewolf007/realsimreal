import torch
import numpy as np
from reward_model.utils.image_util import resize_image, save_image_pkl
from reward_model.action_feasibility_model import RGBDViewReward
from reward_model.offline_reward_model import MultiViewReward
from reward_model.online_reward_model import ask_subtask

class RewardModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.subtask_num = len(cfg.subtask_language_info)
        self.subtask_language_info = cfg.subtask_language_info
        self.subtask_object_info = cfg.subtask_object_info
        self.reward_fn_map = {
            "test": self.test_online_reward,
            "online": self.online_reward,
            "offline": self.offline_reward,
            "action_feasible": self.action_feasible,
        }
        self.last_action = None
        self.last_observation = None
        self.last_info = None
        self.sum_reward = 0
        self.subtask_idx = 0
        self.subtask_pre_flag = False if self.cfg.use_pre_reward else True
        self.subtask_done_flag = False
        self.load_offline_reward_pretrain = False
        self.load_action_feasible_pretrain = False

    def reset(self):
        self.last_action = None
        self.last_observation = None
        self.last_info = None
        self.sum_reward = 0
        self.subtask_idx = 0
        self.subtask_pre_flag = False if self.cfg.use_pre_reward else True
        self.subtask_done_flag = False

    def update(self, action, observation, info):
        if self.subtask_done_flag and self.subtask_pre_flag:
            self.subtask_idx += 1
            self.subtask_pre_flag = False if self.cfg.use_pre_reward else True
            self.subtask_done_flag = False
        self.last_action = action
        self.last_observation = observation
        self.last_info = info

    def save_reward_data(self, image_dict, reward):
        image_dict['result'] = reward
        if reward > 0:
            save_image_pkl(image_dict, self.cfg.agent.online_data_save_path + "/done/", save_ori_image=False)
        else:   
            save_image_pkl(image_dict, self.cfg.agent.online_data_save_path + "/fail/", save_ori_image=False)

    def load_offline_reward_model(self):
        self.load_offline_reward_pretrain = True
        self.offline_reward_model = MultiViewReward()
        self.offline_reward_model.load_model(self.cfg.offline_reward_model_path)
        self.offline_reward_model.eval()

    def load_action_feasible_model(self):
        self.load_action_feasible_pretrain = True
        self.action_feasible_model = RGBDViewReward()
        self.action_feasible_model.load_model(self.cfg.action_feasible_model_path)
        self.action_feasible_model.eval()

    def online_reward(self, observations, action, info, reward = -1, is_save=False, is_train=True):
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.25),
            "right_view": resize_image(observations["rightview_image"], 0.25),
            "bird_view": resize_image(observations["birdview_image"], 0.25),
            "sceneview_depth": observations["sceneview_depth"],
            "sceneview_rgb": observations["sceneview_image"],
        }
        if self.subtask_pre_flag == False:
            given_reward = ask_subtask(image_dict, 
                                       self.subtask_object_info[self.subtask_idx][0], 
                                       self.subtask_object_info[self.subtask_idx][1], 
                                       self.subtask_language_info[self.subtask_idx],
                                       self.subtask_pre_flag)
            if given_reward:
                reward == -1
            else:
                reward = 100
                self.subtask_pre_flag = True
                self.sum_reward += reward
        else:
            given_reward = ask_subtask(image_dict, 
                            self.subtask_object_info[self.subtask_idx][0], 
                            self.subtask_object_info[self.subtask_idx][1], 
                            self.subtask_language_info[self.subtask_idx],
                            self.subtask_pre_flag)
            if given_reward:
                reward == -1
            else:
                reward = 100
                self.subtask_done_flag = True
                self.sum_reward += reward
        reward_info = {self.subtask_language_info[self.subtask_idx]: {"pre_reward": self.subtask_pre_flag, "done_reward": self.subtask_done_flag}}
        self.update(action, observations, info)
        done = True if self.subtask_idx == self.subtask_num else False
        if info["truncation"] == True or done:
            self.reset()
        if is_save:
            self.save_reward_data(image_dict, reward)
        return reward, done, reward_info

    def offline_reward(self, observations, action, info, reward = -1, is_save=False, is_train=True):
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.25),
            "right_view": resize_image(observations["rightview_image"], 0.25),
            "bird_view": resize_image(observations["birdview_image"], 0.25),
            "sceneview_depth": observations["sceneview_depth"],
            "sceneview_rgb": observations["sceneview_image"],
        }
        bird_view_img = image_dict['bird_view'].astype(np.float32) / 255.0
        front_view_img = image_dict['front_view'].astype(np.float32) / 255.0
        right_view_img = image_dict['right_view'].astype(np.float32) / 255.0
        step_image = np.concatenate([resize_image(bird_view_img, target_size=(224, 224)), resize_image(front_view_img, target_size=(224, 224)), resize_image(right_view_img, target_size=(224, 224))], axis=2)
        step_image = np.transpose(step_image, (2, 0, 1))
        tensor_step_image = torch.tensor(step_image).to(self.cfg.device)
        if self.load_offline_reward_pretrain == False:
            self.load_offline_reward_model()
        if self.subtask_pre_flag == False:
            given_reward = self.offline_reward_model.get_reward(tensor_step_image)
            if given_reward:
                reward == -1
            else:
                reward = 100
                self.subtask_pre_flag = True
                self.sum_reward += reward
        else:
            given_reward = self.offline_reward_model.get_reward(tensor_step_image)
            if given_reward:
                reward == -1
            else:
                reward = 100
                self.subtask_done_flag = True
                self.sum_reward += reward
        reward_info = {self.subtask_language_info[self.subtask_idx]: {"pre_reward": self.subtask_pre_flag, "done_reward": self.subtask_done_flag}}
        self.update(action, observations, info)
        done = True if self.subtask_idx == self.subtask_num else False
        if info["truncation"] == True or done:
            self.reset()
        if is_save:
            self.save_reward_data(image_dict, reward)
        return reward, done, reward_info

    def action_feasible(self, observations, action, info, reward = -1, is_save=False, is_train=True):
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.25),
            "right_view": resize_image(observations["rightview_image"], 0.25),
            "bird_view": resize_image(observations["birdview_image"], 0.25),
            "sceneview_depth": observations["sceneview_depth"],
            "sceneview_rgb": observations["sceneview_image"],
        }
        scene_rgb_img = image_dict['sceneview_depth'].astype(np.float32) / 255.0
        scene_depth_img = image_dict['sceneview_rgb'].astype(np.float32)
        step_image = np.concatenate([scene_rgb_img, scene_depth_img], axis=2)
        step_image = np.transpose(step_image, (2, 0, 1))
        tensor_step_image = torch.tensor(step_image).to(self.cfg.device)
        if self.load_action_feasible_pretrain == False:
            self.load_action_feasible_model()
        if self.subtask_pre_flag == False:
            given_reward = self.action_feasible_model.get_reward(tensor_step_image)
            if given_reward:
                reward == -1
            else:
                reward = 100
                self.subtask_pre_flag = True
                self.sum_reward += reward
        else:
            given_reward = self.action_feasible_model.get_reward(tensor_step_image)
            if given_reward:
                reward == -1
            else:
                reward = 100
                self.subtask_done_flag = True
                self.sum_reward += reward
        reward_info = {self.subtask_language_info[self.subtask_idx]: {"pre_reward": self.subtask_pre_flag, "done_reward": self.subtask_done_flag}}
        self.update(action, observations, info)
        done = True if self.subtask_idx == self.subtask_num else False
        if info["truncation"] == True or done:
            self.reset()
        if is_save:
            self.save_reward_data(image_dict, reward)
        return reward, done, reward_info

    def test_online_reward(self, observations, action, info, reward = -1, is_save=False, is_train=True):
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.25),
            "right_view": resize_image(observations["rightview_image"], 0.25),
            "bird_view": resize_image(observations["birdview_image"], 0.25),
            "sceneview_depth": observations["sceneview_depth"],
            "sceneview_rgb": observations["sceneview_image"],
        }
        if self.subtask_pre_flag == False:
            # pre_reward function
            if reward == -1:
                pass
            else:
                self.subtask_pre_flag = True
                self.sum_reward += reward
        else:
            # done_reward function
            if reward == -1:
                pass
            else:
                self.subtask_done_flag = True
                self.sum_reward += reward
        reward_info = {self.subtask_language_info[self.subtask_idx]: {"pre_reward": self.subtask_pre_flag, "done_reward": self.subtask_done_flag}}
        self.update(action, observations, info)
        done = True if self.subtask_idx == self.subtask_num else False
        if info["truncation"] == True or done:
            self.reset()
        if is_save:
            self.save_reward_data(image_dict, reward)
        return reward, done, reward_info

    def __call__(self, observations, action, info, reward = -1, reward_type="test", is_save=False, is_train=True):
        reward_fn = self.reward_fn_map[reward_type]
        return reward_fn(observations, action, info, reward, is_save, is_train)