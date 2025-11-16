import os
import hydra
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
class SimToReal:
    def __init__(self, cfg, action_shape):
        self.agent_name = cfg.agent_name
        self.cfg = cfg
        self.action_shape = action_shape
        self.xyz_mean, self.xyz_std = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        self.device = "cuda"
        self.server_url = "http://10.184.17.177:8000/"
        self.pose_url_suffix = "act"
        self.init_policy()
        self.img_post_process_fn = make_imageprocess_fn(cfg)
        self.action_pre_process_fn = make_actionprocess_fn(cfg)
        self.step = 0


    def init_policy(self):
        self.test_agent = make_policy_agent(self.cfg, self.cfg.agent_name, self.cfg.device, self.action_shape,
                                            is_train=False)

    def get_action(self, obs):
        # processed_obs = self.img_post_process_fn(obs)
        if self.agent_name == "LaNE":
            raise NotImplementedError
        elif self.agent_name == "maniwhere":
            processed_obs = np.random.rand(10, 128, 128).astype(np.float32)
            action = self.test_agent.get_action(processed_obs, self.step)
        else:
            raise NotImplementedError
        self.step += 1
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


@hydra.main(config_path='configs/pick_maniwhere.yaml', strict=True)
def main_policy(cfg):
    model_dir = "/home/haowen/hw_mine/Real_Sim_Real/experiments/best_snapshot_740000.pt"
    cfg.agent["model_dir"] = model_dir
    action_shape = (7,)
    real_agent = SimToReal(cfg, action_shape)
    obs = np.random.rand(1536, 2048, 4)
    action = real_agent.get_action(obs)
    print("action", action)

if __name__ == "__main__":
    main_policy()
