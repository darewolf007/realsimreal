import hydra
import os
import numpy as np
from omegaconf import OmegaConf
from utils.image_util import resize_image
from agent_policy.base_agent_policy import BaseAgentPolicy
from simple_sim.environment.pick_environment import PickBananaSimulation

def make_postprocess_fn(cfg):
    if cfg.agent_name == "LaNE":
        def img_postprocess_fn(obs):
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
    if env_name=="PickBanana":
        env = PickBananaSimulation("UR5e",
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
    else:
        raise NotImplementedError
    return env

@hydra.main(config_path='configs/banana_lane.yaml', strict=True)
def train_policy(cfg):
    env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    test_env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    obs_shape = (3 * len(cfg.agent.cameras) * cfg.agent.frame_stack, cfg.agent.image_size, cfg.agent.image_size) 
    action_shape = env.action_space.shape
    agent = make_policy_agent(cfg.agent, cfg.agent_name, cfg.device, obs_shape, action_shape, is_train=True)
    agent.train_agent(env, test_env, make_postprocess_fn(cfg))

if __name__ == "__main__":
    train_policy()