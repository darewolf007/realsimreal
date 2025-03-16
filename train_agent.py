import hydra
import os
import numpy as np
from omegaconf import OmegaConf
from utils.image_util import resize_image, save_image_pkl
from agent_policy.base_agent_policy import BaseAgentPolicy
from simple_sim.environment.pick_environment import PickBananaSimulation
from reward_model.reward import RewardModel

ENV_DICT = {
    "PickBanana": PickBananaSimulation}

def make_actionprocess_fn(cfg):
    if cfg.agent_name == "LaNE":
        def action_postprocess_fn(action, action_mean = None, action_std = None):
            pocess_action = np.copy(action)
            pocess_action[:3] =  action[:3]* action_std[0] + action_mean[0]
            pocess_action[:3] = np.clip(pocess_action[:3], -cfg.env_info['max_action'], cfg.env_info['max_action'])
            pocess_action[:3] = pocess_action[:3]/100
            pocess_action[-1] = 1 if pocess_action[-1] > 0 else -1
            return pocess_action
        return action_postprocess_fn

def make_imageprocess_fn(cfg):
    view_name = cfg.train_camera_name + "_image"
    if cfg.agent_name == "LaNE":
        def img_postprocess_fn(observations):
            obs = observations[view_name]
            obs = resize_image(obs, 1/12)
            obs = np.transpose(obs, (2, 0, 1))
            return obs
        return img_postprocess_fn
    

def make_policy_agent(agent_info, agent_name, device, obs_shape, action_shape, is_train):
    base_path = os.path.dirname(os.path.abspath(__file__))
    if agent_name == "LaNE":
        agent_info['replay_buffer_load_dir'] = os.path.join(base_path, agent_info['replay_buffer_load_dir'])
        agent_info['work_dir'] = os.path.join(base_path, agent_info['work_dir'])
        agent = BaseAgentPolicy(agent_info, agent_name, device, obs_shape, action_shape, is_train)
        return agent
    else:
        raise NotImplementedError

def make_env(env_name, env_info):
    base_path = os.path.dirname(os.path.abspath(__file__))
    env_info['hand_eye_path'] = os.path.join(base_path, env_info['hand_eye_path'])
    env_info['replay_buffer_load_dir'] = os.path.join(base_path, env_info['replay_buffer_load_dir'])
    env = ENV_DICT[env_name]
    return env("UR5e",
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

@hydra.main(config_path='configs/banana_lane.yaml', strict=True)
def train_policy(cfg):
    env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    env.replay(img_post_process_fn=make_imageprocess_fn(cfg), reward_fn=RewardModel(cfg), action_pre_process_fn=make_actionprocess_fn(cfg))
    test_env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    obs_shape = (3 * len(cfg.agent.cameras) * cfg.agent.frame_stack, cfg.agent.image_size, cfg.agent.image_size) 
    action_shape = env.action_space.shape
    agent = make_policy_agent(cfg.agent, cfg.agent_name, cfg.device, obs_shape, action_shape, is_train=True)
    agent.train_agent(env, test_env, img_post_process_fn=make_imageprocess_fn(cfg), reward_fn=RewardModel(cfg), action_pre_process_fn=make_actionprocess_fn(cfg))

if __name__ == "__main__":
    train_policy()