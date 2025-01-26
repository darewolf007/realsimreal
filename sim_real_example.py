import os
import torch
import numpy as np
from agent_policy.few_shot_RL.bc_sac_policy import BCSACPolicy
import requests
import numpy as np
import io
import cv2
import time
def set_params(task_name, promot_list = None):
    policy_params = {
    "work_dir": None,
    "task_name": task_name,
    "sub_task_promot": promot_list,
    "replay_buffer_capacity": 300000,
    "replay_buffer_load_dir": None,
    "replay_buffer_keep_loaded": True,
    "pretrain_mode": None,
    "pre_transform_image_size": 128,
    "image_size": 112,
    "cameras": [0],
    "observation_type": "pixel",
    "reward_type": "sparse",
    "control": None,
    "action_repeat": 1,
    "frame_stack": 1,
    "num_updates": 1,
    "model_dir": None,
    "model_step": 40000,
    "agent_name": "rad_sac",
    "init_steps": 2000,
    "num_train_steps": 250000,
    "bc_train_steps": 20,
    "batch_size": 128,
    "hidden_dim": 1024,
    "eval_freq": 1000,
    "num_eval_episodes": 2,
    "critic_lr": 0.001,
    "critic_beta": 0.9,
    "critic_tau": 0.01,
    "critic_target_update_freq": 2,
    "actor_lr": 0.001,
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
    "save_tb": False,
    "save_buffer": False,
    "save_video": False,
    "save_sac": False,
    "detach_encoder": False,
    "v_clip_low": -100,
    "v_clip_high": 100,
    "action_noise": None,
    "final_demo_density": 0.5,
    "data_augs": "center_crop",
    "log_interval": 200,
    "conv_layer_norm": True,
    "p_reward": 1,
    }
    return policy_params

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
    def __init__(self, agent_name = "rad_sac", policy_param = None, task_max_step = 100, model_dir = None):
        self.agent_name = agent_name
        self.policy_param = policy_param
        self.max_step = task_max_step
        self.device = "cuda"
        self.server_url = "http://10.184.17.177:8000/"
        self.pose_url_suffix = "act"
        self.init_policy(model_dir)
        self.previous_gripper_action = None

    def init_policy(self, model_dir):
        if self.agent_name == "rad_sac":
            self.policy = BCSACPolicy(env = None, device = self.device, params = self.policy_param)
            self.policy_agent = self.policy.init_policy(self.agent_name, (7,), model_dir)

    def get_action(self, obs):
        obs = self.preprocess_obs(obs)
        action = self.policy.get_action(self.policy_agent, obs)
        return action
    
    def preprocess_obs(self, obs):
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
    
    
if __name__ == "__main__":
    # response = requests.post("http://10.184.17.177:8000/vla")
    # print(response)
    task_name = "Pick up banana"
    agent_name = "rad_sac"
    model_dir = "/home/haowen/hw_mine/Real_Sim_Real/results/pick_banana"
    policy_params = set_params(task_name, [task_name])
    sim_real_example = SimRealExample(agent_name, policy_params, task_max_step=60, model_dir=model_dir)
    # obs = np.random.rand(3, 112, 112)
    # action = sim_real_example.get_action(obs)
    obs = np.random.rand(1256, 1556, 3)
    action = sim_real_example.send_online_data(obs, "pick up banana")
    print("action", action)
