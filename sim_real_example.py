import os
import torch
import numpy as np
from agent_policy.few_shot_RL.bc_sac_policy import BCSACPolicy

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

class SimRealExample:
    def __init__(self, agent_name = "rad_sac", policy_param = None, task_max_step = 100, model_dir = None):
        self.agent_name = agent_name
        self.policy_param = policy_param
        self.max_step = task_max_step
        self.device = "cuda"
        self.init_policy(model_dir)

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
        return action
    
    
if __name__ == "__main__":
    task_name = "Pick up banana"
    agent_name = "rad_sac"
    model_dir = "/home/haowen/hw_mine/Real_Sim_Real/results/pick_banana"
    policy_params = set_params(task_name, [task_name])
    sim_real_example = SimRealExample(agent_name, policy_params, task_max_step=60, model_dir=model_dir)
    obs = np.random.rand(3, 112, 112)
    action = sim_real_example.get_action(obs)
    print("action", action)