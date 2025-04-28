import hydra
import os
import numpy as np
from omegaconf import OmegaConf
from utils.image_util import resize_image, save_image_pkl
from agent_policy.agent_policy import BaseAgentPolicy
from simple_sim.environment.pick_environment import PickBananaSimulation
from simple_sim.environment.pick_place_environment import PickApplePlaceBowlSimulation
from reward_model.reward import RewardModel
from reward_model.lane_reward_model import DINOE2CSacAgent as LaNERewardModel

ENV_DICT = {
    "PickBanana": PickBananaSimulation,
    "PickApplePlaceBowl":PickApplePlaceBowlSimulation}
# add PourCan, PourCheeze, StackCan, InsertMarker
def make_reward_fn(cfg, action_shape = None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    if cfg.agent_name == "LaNE" or cfg.agent_name == "mine":
        return RewardModel(cfg, base_path=os.path.dirname(os.path.abspath(__file__)))
    elif cfg.agent_name == "maniwhere":
        replay_buffer_load_dir = os.path.join(base_path, cfg['replay_buffer_load_dir'])
        reward_obs_shape = (3 * len(cfg.reward_camera_name), cfg.agent.image_size, cfg.agent.image_size)
        device = cfg.device
        action_shape = action_shape
        reward_model_dir = os.path.join(base_path, cfg['reward_model_dir'])
        dino_model_dir = os.path.join(base_path, cfg['offline_dino_dir'])
        reward_model = LaNERewardModel(replay_buffer_load_dir, reward_obs_shape, action_shape, device, reward_model_dir, dino_model_dir)
        return reward_model
    else:
        raise NotImplementedError

def make_actionprocess_fn(cfg):
    if cfg.agent_name == "LaNE" or cfg.agent_name == "mine":
        def action_postprocess_fn(action, action_mean = None, action_std = None):
            pocess_action = np.copy(action)
            pocess_action[:3] =  action[:3]* action_std + action_mean
            pocess_action[:3] = np.clip(pocess_action[:3], -cfg.env_info['max_action'], cfg.env_info['max_action'])
            pocess_action[:3] = pocess_action[:3]/100
            pocess_action[-1] = 1 if pocess_action[-1] > 0 else -1
            return pocess_action
        return action_postprocess_fn
    elif cfg.agent_name == "maniwhere":
        # action_mean = cfg.agent.action_mean
        # action_std = cfg.agent.action_std
        def action_postprocess_fn(action, action_mean = None, action_std = None):
            if action_mean is None or action_std is None:
                action_mean = 0
                action_std = 1
            pocess_action = np.copy(action)
            pocess_action[:3] =  action[:3]* action_std + action_mean
            pocess_action[:3] = np.clip(pocess_action[:3], -cfg.env_info['max_action'], cfg.env_info['max_action'])
            pocess_action[:3] = pocess_action[:3]/100
            pocess_action[-1] = 1 if pocess_action[-1] > 0 else -1
            return pocess_action
        return action_postprocess_fn
    else:
        raise NotImplementedError

def make_imageprocess_fn(cfg):
    view_name = cfg.train_camera_name + "_image"
    if cfg.agent_name == "LaNE":
        def img_postprocess_fn(observations):
            obs = observations[view_name]
            obs = resize_image(obs, 1/12)
            obs = np.transpose(obs, (2, 0, 1))
            return obs
        return img_postprocess_fn
    elif cfg.agent_name == "mine":
        view_name_list = [view_name + "_image" for view_name in cfg.reward_camera_name]
        def img_postprocess_fn(observations):
            obs_list = []
            for view_name in view_name_list:
                obs = observations[view_name]
                obs = resize_image(obs, 1/12)
                obs = np.transpose(obs, (2, 0, 1))
                obs_list.append(obs)
            return np.concatenate(obs_list, axis=0)
        return img_postprocess_fn
    elif cfg.agent_name == "maniwhere":
        from agent_policy.maniwhere.utils import ExtendedTimeStep, CameraViewWrapper
        from dm_env import StepType, specs
        train_camera_view = [view_name + "_image" for view_name in cfg.agent.train_camera_name]
        train_camera_view_wrapper = [CameraViewWrapper(num_frames=cfg.frame_stack, height=cfg.img_size, width=cfg.img_size, depth=cfg.use_depth) for _ in range(len(cfg.agent.train_camera_name))]
        eval_camera_view_wrapper = CameraViewWrapper(num_frames=cfg.frame_stack, height=cfg.img_size, width=cfg.img_size, depth=cfg.use_depth)
        def img_postprocess_fn(obs, reward, info, action, is_reset, is_train):
            if is_train:
                frame_stack_obs = []
                for i in range(len(cfg.agent.train_camera_name)):
                    obs_image = obs[train_camera_view[i]]
                    obs_image = resize_image(obs_image, target_size=(cfg.img_size, cfg.img_size))
                    if obs_image.shape[-1] == 3:
                        obs_image = np.transpose(obs_image, (2, 0, 1))
                    if is_reset:
                        frame_stack_obs.append(train_camera_view_wrapper[i].reset(obs_image))
                    else:
                        frame_stack_obs.append(train_camera_view_wrapper[i].step(obs_image))
                frame_stack_obs = np.concatenate(frame_stack_obs, axis=0)
            else:
                obs_image = obs[train_camera_view[0]]
                obs_image = resize_image(obs_image, target_size=(cfg.img_size, cfg.img_size))
                if obs_image.shape[-1] == 3:
                    obs_image = np.transpose(obs_image, (2, 0, 1))
                if is_reset:
                    frame_stack_obs =eval_camera_view_wrapper.reset(obs_image)
                else:
                    frame_stack_obs =eval_camera_view_wrapper.step(obs_image)
            if is_reset:
                step_type = StepType.FIRST
            elif info["truncation"] == True:
                step_type = StepType.LAST
            else:
                step_type = StepType.MID
            return ExtendedTimeStep(observation=frame_stack_obs,
                        step_type=step_type,
                        action=action,
                        reward=reward or 0.0,
                        discount=1.0,
                        not_done = (not info["truncation"]),)
        return img_postprocess_fn
    else:
        raise NotImplementedError
    
def make_policy_agent(cfg, agent_name, device, obs_shape, action_shape, is_train):
    agent_info = cfg.agent if not cfg.is_finetuning else cfg.finetuning
    base_path = os.path.dirname(os.path.abspath(__file__))
    if agent_name == "LaNE":
        agent_info['replay_buffer_load_dir'] = os.path.join(base_path, agent_info['replay_buffer_load_dir'])
        agent = BaseAgentPolicy(agent_info, agent_name, device, obs_shape, action_shape, is_train)
        return agent
    elif agent_name == "mine":
        agent_info['cameras'] = [cam_id for cam_id in range(len(cfg.reward_camera_name))]
        agent_info['replay_buffer_load_dir'] = os.path.join(base_path, agent_info['replay_buffer_load_dir'])
        agent = BaseAgentPolicy(agent_info, agent_name, device, obs_shape, action_shape, is_train)
        return agent
    elif agent_name == "maniwhere":
        agent = BaseAgentPolicy(cfg, agent_name, device, obs_shape, action_shape, is_train)
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
        camera_names=env_info['camera_names'],
        schedule_random = env_info['schedule_random'],)

def eval_agent_in_env(cfg):
    cfg.agent["model_dir"] = "/home/haowen/hw_mine/Real_Sim_Real/experiments/sparse-dino_e2c_sac-pixel-crop-02-18-pick up banana-pick up banana30-im112-b128-nu1-s1-id56720/model"
    cfg.env_info['has_renderer'] = True
    eval_env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    obs_shape = (3 * len(cfg.agent.cameras) * cfg.agent.frame_stack, cfg.agent.image_size, cfg.agent.image_size) 
    action_shape = eval_env.action_space.shape
    test_agent = make_policy_agent(cfg, cfg.agent_name, cfg.device, obs_shape, action_shape, is_train=False)
    img_post_process_fn = make_imageprocess_fn(cfg)
    reward_fn = RewardModel(cfg, base_path=os.path.dirname(os.path.abspath(__file__)))
    action_pre_process_fn = make_actionprocess_fn(cfg)
    obs = eval_env.reset()
    xyz_mean, xyz_std = eval_env.action_info()
    prev_gripper_state = 0
    for _ in range(50):
        new_obs = img_post_process_fn(obs)
        action = test_agent.get_action(new_obs)
        action = action_pre_process_fn(action, xyz_mean, xyz_std)
        pre_reward, _, pre_reward_info = reward_fn(obs, action, None, -1, reward_type="test", is_save=False, is_train=False)
        print(pre_reward)
        if action[-1] != prev_gripper_state:
            if not list(pre_reward_info.values())[0]["pre_reward"]:
                print("Action is not feasible")
                continue
        obs, reward, done, info = eval_env.step(action)
        prev_gripper_state = action[-1]
        done_reward, _, done_reward_info = reward_fn(obs, action, None, -1, reward_type="test", is_save=False, is_train=False)
        print(done_reward)
        if done:
            obs = eval_env.reset()

# @hydra.main(config_path='configs/pick_lane.yaml', strict=True)
# @hydra.main(config_path='configs/pickplace_lane.yaml', strict=True)
# @hydra.main(config_path='configs/pickplace_mine.yaml', strict=True)
# @hydra.main(config_path='configs/pickplace_maniwhere.yaml', strict=True)
@hydra.main(config_path='configs/pick_maniwhere.yaml', strict=True)
def train_policy(cfg):
    env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    # env.replay(img_post_process_fn=make_imageprocess_fn(cfg), reward_fn=RewardModel(cfg, base_path=os.path.dirname(os.path.abspath(__file__))), action_pre_process_fn=make_actionprocess_fn(cfg))
    test_env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    obs_shape = (3 * len(cfg.agent.cameras) * cfg.agent.frame_stack, cfg.agent.image_size, cfg.agent.image_size) 
    action_shape = env.action_space.shape
    agent = make_policy_agent(cfg, cfg.agent_name, cfg.device, obs_shape, action_shape, is_train=True)
    agent.train_agent(env, test_env, img_post_process_fn=make_imageprocess_fn(cfg), reward_fn=make_reward_fn(cfg, action_shape), action_pre_process_fn=make_actionprocess_fn(cfg))

if __name__ == "__main__":
    train_policy()