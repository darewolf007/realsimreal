import os
import torch
import numpy as np
import requests
import numpy as np
import io
import cv2
import time
from omegaconf import OmegaConf
from train_agent import make_policy_agent, make_imageprocess_fn, make_actionprocess_fn
from reward_model.reward import RewardModel
from utils.image_util import resize_image


def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = np.array([w, x, y, z])
    return quat

class SimRealExample:
    def __init__(self, agent_name = "LaNE", policy_param = None, task_max_step = 100, model_dir = None):
        self.agent_name = agent_name
        self.policy_param = policy_param
        self.action_shape = (7, )
        self.xyz_mean, self.xyz_std = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        self.max_step = task_max_step
        self.device = "cuda"
        self.server_url = "http://10.184.17.177:8000/"
        self.pose_url_suffix = "act"
        self.init_policy(model_dir)
        self.previous_gripper_action = None

    def init_policy(self, model_dir):
        if self.agent_name == "LaNE":
            self.policy_param.agent["model_dir"] = model_dir
            obs_shape = (3 * len(self.policy_param.agent.cameras) * self.policy_param.agent.frame_stack, self.policy_param.agent.image_size, self.policy_param.agent.image_size) 
            self.agent = make_policy_agent(self.policy_param.agent, self.policy_param.agent_name, self.policy_param.device, obs_shape, self.action_shape, is_train=False)
            self.reward_fn = RewardModel(self.policy_param, base_path=os.path.dirname(os.path.abspath(__file__)))
            self.action_pre_process_fn = make_actionprocess_fn(self.policy_param)
        else:
            raise NotImplementedError

    def get_action(self, obs):
        ori_obs = obs.copy()
        obs = self.preprocess_obs(obs[:, :, :3])
        if self.agent_name == "LaNE":
            action, done = self.get_LaNE_action(obs, ori_obs)
        else:
            raise NotImplementedError
        return action, done
    
    def preprocess_obs(self, obs):
        if self.agent_name == "LaNE":
            obs = resize_image(obs, 1/12)
            obs = np.transpose(obs, (2, 0, 1))
            torch_obs = torch.from_numpy(obs).to('cpu').float()
        else:
            torch_obs = torch.from_numpy(obs).to('cpu').float()
        return torch_obs

    def postprocess_action(self, action):
        current_gripper_action = action[-1]
        if self.previous_gripper_action is None:
            relative_gripper_action = np.array([0])
        else:
            relative_gripper_action = (
                self.previous_gripper_action - current_gripper_action
            )  # google robot 1 = close; -1 = open
        self.previous_gripper_action = current_gripper_action

        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            gripper_action = 1
        else:
            gripper_action = 0
        world_vector = action[:3]
        action_rotation_delta = action[3:]
        quat = euler_to_quaternion(action_rotation_delta[0], action_rotation_delta[1], action_rotation_delta[2])
        if gripper_action < 0.5:
            gripper_action = 0
        else:
            gripper_action = 1
        action = np.concatenate([world_vector, quat, [gripper_action]])
        return action
    
    def send_online_data(self, obs, task_name):
        url = self.server_url + self.pose_url_suffix
        np_rgb_image = np.array(obs)
        _, image_encoded = cv2.imencode('.jpg', np_rgb_image)
        image_bytes = io.BytesIO(image_encoded.tobytes())
        files = {
            "image_file": ("image.jpg", image_bytes, "image/jpeg"),
        }
        data = {
            "label": task_name,
        }
        response = requests.post("http://10.184.17.177:8000/vla",files=files ,data=data)
        if response.status_code == 200:
            task_info = response.json()
            print(f"Task started: {task_info}")
            return self.postprocess_action(task_info["action"])
        else:
            print("Failed to get a valid response from server. Status code:", response.status_code)
    
    def get_LaNE_action(self, obs, ori_obs):
        action = self.agent.get_action(obs)
        action = self.action_pre_process_fn(action, self.xyz_mean, self.xyz_std)
        image_dict = {
            "sceneview_depth": ori_obs[:,:,3],
            "sceneview_image": ori_obs[:,:,:3],
        }
        pre_reward, _, pre_reward_info = self.reward_fn(image_dict, action, {"truncation": False}, -1, reward_type="action_feasible", is_save=False, is_train=False)
        if action[-1] != self.previous_gripper_action:
            if not list(pre_reward_info.values())[0]["pre_reward"]:
                action = None
            else:
                self.previous_gripper_action = action[-1]
                done_reward, _, done_reward_info = self.reward_fn(image_dict, action, {"truncation": False}, -1, reward_type="action_feasible", is_save=False, is_train=False)
                done = True if done_reward > 0 else False
        return action, done
    
if __name__ == "__main__":
    agent_name = "LaNE"
    model_dir = "/home/haowen/hw_mine/Real_Sim_Real/experiments/sparse-dino_e2c_sac-pixel-crop-02-18-pick up banana-pick up banana30-im112-b128-nu1-s1-id56720/model"
    policy_params = OmegaConf.load("/home/haowen/hw_mine/Real_Sim_Real/experiments/Pick up banana/pick up banana-PickBanana-LaNE-test-2025-03-18-16-56-52/.hydra/config.yaml")
    policy_params.agent['reward_model_type'] = "action_feasible"
    policy_params.action_feasible_model_path = "./experiments/action_feasible_model/Pick up banana/best_reward_model.pth"
    sim_real_example = SimRealExample(agent_name, policy_params, task_max_step=60, model_dir=model_dir)
    obs = np.random.rand(1536, 2048, 4)
    action, done = sim_real_example.get_action(obs)
    print("action", action)
    print("done", done)
    obs = np.random.rand(1256, 1556, 3)
    action = sim_real_example.send_online_data(obs, "pick up banana")
    print("action", action)
