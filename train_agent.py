import hydra
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from utils.data_convert_pt import convert_pt_data
from utils.image_util import resize_image, save_image_pkl
from agent_policy.agent_policy import BaseAgentPolicy
from agent_policy.online_vla.pi0_client import Pi0Client
from agent_policy.online_vla.vla_adapter_client import VlaAdapterClient
from simple_sim.environment.pick_environment import LiftBananaSimulation
from simple_sim.environment.pick_place_environment import PickApplePlaceBowlSimulation
from simple_sim.environment.pour_environment import PourCanSimulation
from simple_sim.environment.insert_environment import InsertMarkerSimulation
from simple_sim.environment.stack_environment import StackCanSimulation
from simple_sim.environment.press_environment import PressButtonSimulation
from reward_model.reward import RewardModel
from reward_model.lane_reward_model import DINOE2CSacAgent as LaNERewardModel
# os.environ["WANDB_MODE"] = "offline"

ENV_DICT = {
    "PickBanana": LiftBananaSimulation,
    "PickApplePlaceBowl":PickApplePlaceBowlSimulation,
    "PourCan": PourCanSimulation,
    "StackCan": StackCanSimulation,
    "InsertMarker": InsertMarkerSimulation,
    "PressButton": PressButtonSimulation,}

def make_reward_fn(cfg, action_shape = None):
    if cfg.env_info.reward_type == "dense":
        return None
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
    elif cfg.agent_name == "pi0" or cfg.agent_name == "vla-adapter":
        return None
    elif cfg.agent_name == "ppo":
        return None
    else:
        raise NotImplementedError

def make_actionprocess_fn(cfg):
    if cfg.agent_name == "LaNE" or cfg.agent_name == "mine":
        def action_postprocess_fn(action, action_mean = None, action_std = None):
            pocess_action = np.copy(action)
            pocess_action[:3] = action[:3]* action_std + action_mean
            pocess_action[:3] = np.clip(pocess_action[:3], -cfg.env_info['max_action'], cfg.env_info['max_action'])
            pocess_action[:3] = pocess_action[:3]/100
            pocess_action[-1] = 1 if pocess_action[-1] > 0 else -1
            return pocess_action
        return action_postprocess_fn
    elif cfg.agent_name == "maniwhere":
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
    elif cfg.agent_name == "pi0":
        def action_postprocess_fn(action, action_mean = None, action_std = None):
            pocess_action = np.copy(action)
            pocess_action[-1] = 1 if pocess_action[-1] > 0.5 else -1
            return pocess_action
        return action_postprocess_fn
    elif cfg.agent_name == "vla-adapter":
        from scipy.spatial.transform import Rotation as R
        def action_postprocess_fn(action, action_mean = None, action_std = None):
            pocess_action = np.copy(action)
            pocess_action[-1] = 1 if pocess_action[-1] > 0.5 else -1
            # rotation_action = pocess_action[3:-1]
            # rot = R.from_euler('xyz', rotation_action, degrees=False)
            # quat_action = rot.as_quat()[[3,0,1,2]]
            # quat_pocess_action = np.concatenate([pocess_action[:3], quat_action, pocess_action[-1:]], axis=0)
            return pocess_action
        return action_postprocess_fn
    elif cfg.agent_name == "ppo":
        def action_postprocess_fn(action, action_mean=None, action_std=None):
            if isinstance(action, torch.Tensor):
                raw_action = action.detach().cpu().numpy()[0].copy()
            else:
                raw_action = action.copy()
            pos_limit = 0.005
            process_action = raw_action.copy()
            process_action[:3] = np.tanh(raw_action[:3]) * pos_limit
            process_action[3:-1] = 0.0
            process_action[-1] = 1.0 if raw_action[-1] > 0 else -1.0
            return process_action
        return action_postprocess_fn
    else:
        raise NotImplementedError

def make_imageprocess_fn(cfg):
    if isinstance(cfg.train_camera_name, str):
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
    elif cfg.agent_name == "pi0":
        def img_postprocess_fn(observations):
            vla_info = {}
            vla_info['scene_image'] = resize_image(observations['sceneview_image'], 1/12)
            vla_info['hand_image'] = resize_image(observations['robot0_eye_in_hand_image'], 1/12)
            vla_info['joint'] = observations['robot0_joint_observation']
            vla_info['gripper'] = observations['robot0_gripper_observation']
            vla_info['eff'] = observations['robot0_eff_observation']
            vla_info['task_prompt'] = cfg['task_name']
            return vla_info
        return img_postprocess_fn
    elif cfg.agent_name == "vla-adapter":
        def img_postprocess_fn(observations):
            vla_info = {}
            vla_info['scene_image'] = resize_image(observations['sceneview_image'], 1/12)
            vla_info['hand_image'] = resize_image(observations['robot0_eye_in_hand_image'], 1/12)
            vla_info['joint'] = observations['robot0_joint_observation']
            vla_info['gripper'] = observations['robot0_gripper_observation']
            translation_eff = observations['robot0_eff_observation'][:3]
            rotation_eff = observations['robot0_eff_observation'][3:-1][[1,2,3,0]]
            eff_observation = np.concatenate([translation_eff, rotation_eff, np.array([observations['robot0_eff_observation'][-1]])], axis=0)
            vla_info['eff'] = eff_observation
            vla_info['task_prompt'] = cfg['task_name']
            return vla_info
        return img_postprocess_fn
    elif cfg.agent_name == "ppo":
        def img_postprocess_fn(observations):
            obs_dict = {}
            for train_camera_name in cfg.agent.train_camera_name:
                obs = observations[train_camera_name+"_image"]
                obs = resize_image(obs, 1/12)[None, ...]
                obs_dict[train_camera_name] = torch.from_numpy(obs).to(cfg.device)
            # obs_dict = {"rgb": torch.from_numpy(obs).to(cfg.device),
            #             }
            return obs_dict
        return img_postprocess_fn
    else:
        raise NotImplementedError
    
def make_policy_agent(cfg, agent_name, device, action_shape, is_train):
    base_path = os.path.dirname(os.path.abspath(__file__))
    if agent_name == "LaNE":
        agent_info = cfg.agent if not cfg.is_finetuning else cfg.finetuning
        obs_shape = (3 * len(cfg.agent.cameras) * cfg.agent.frame_stack, cfg.agent.image_size, cfg.agent.image_size)
        agent_info['replay_buffer_load_dir'] = os.path.join(base_path, agent_info['replay_buffer_load_dir'])
        agent = BaseAgentPolicy(agent_info, agent_name, device, obs_shape, action_shape, is_train)
        return agent
    elif agent_name == "mine":
        agent_info = cfg.agent if not cfg.is_finetuning else cfg.finetuning
        obs_shape = (3 * len(cfg.agent.cameras) * cfg.agent.frame_stack, cfg.agent.image_size, cfg.agent.image_size)
        agent_info['cameras'] = [cam_id for cam_id in range(len(cfg.reward_camera_name))]
        agent_info['replay_buffer_load_dir'] = os.path.join(base_path, agent_info['replay_buffer_load_dir'])
        agent = BaseAgentPolicy(agent_info, agent_name, device, obs_shape, action_shape, is_train)
        return agent
    elif agent_name == "maniwhere":
        obs_shape = (3 * len(cfg.agent.cameras) * cfg.agent.frame_stack, cfg.agent.image_size, cfg.agent.image_size)
        agent = BaseAgentPolicy(cfg, agent_name, device, obs_shape, action_shape, is_train)
        return agent
    elif agent_name == "pi0":
        agent = Pi0Client()
        return agent
    elif agent_name == "octo":
        pass
    elif agent_name == "vla-adapter":
        agent = VlaAdapterClient()
        return agent
    elif agent_name == "ppo":
        agent_info = cfg.agent
        agent_info['save_path'] = os.path.join(base_path, cfg['save_dir'])
        obs_shape = (cfg.agent.image_size, cfg.agent.image_size, 3)
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
    cfg.env_info['has_renderer'] = True
    eval_env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    eval_env.reset()
    # cfg.agent["model_dir"] = None
    # xyz_mean, xyz_std = eval_env.action_info() # vla not use
    xyz_mean, xyz_std = 0, 1
    test_agent = make_policy_agent(cfg, cfg.agent_name, cfg.device, eval_env.action_space.shape, is_train=False)
    img_post_process_fn = make_imageprocess_fn(cfg)
    action_pre_process_fn = make_actionprocess_fn(cfg)
    eval_time = 1
    for _ in range(eval_time):
        obs = eval_env.reset()
        while True:
            new_obs = img_post_process_fn(obs)
            action = test_agent.get_action(new_obs)
            if action.ndim == 1:
                action = action[None, :]
            for i in range(action.shape[0]):
                action = action_pre_process_fn(action[i], xyz_mean, xyz_std)
                obs, reward, done, info = eval_env.step(action)
            if done or info.get("truncation", False):
                break

# @hydra.main(config_path='configs/pick_lane.yaml', strict=True)
# @hydra.main(config_path='configs/pickplace_lane.yaml', strict=True)
# @hydra.main(config_path='configs/pickplace_mine.yaml', strict=True)
# @hydra.main(config_path='configs/pickplace_maniwhere.yaml', strict=True)
# @hydra.main(config_path='configs/pick_maniwhere.yaml', strict=True)
# @hydra.main(config_path='configs/pick_vla.yaml', strict=True)
# @hydra.main(config_path='configs/pour_vla.yaml', strict=True)
# @hydra.main(config_path='configs/lift_maniwhere.yaml', strict=True)
@hydra.main(config_path='configs/lift_ppo.yaml', strict=True)
def main_policy(cfg):
    # convert_pt_data(cfg.task_name.replace(" ", "_"), cfg.dataset_name, os.path.dirname(os.path.abspath(__file__)))
    env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
    test_env = False
    if test_env:
        import cv2
        for _ in range(10):
            obs = env.reset()
            img_post_process_fn = make_imageprocess_fn(cfg)
            new_obs = img_post_process_fn(obs)
            cv2.imshow("replay_obs", new_obs['rgb'][0].cpu().numpy()[:, :, ::-1])
            cv2.waitKey(10)
        cv2.destroyAllWindows()
    else:
        if cfg.mode == "eval":
            eval_agent_in_env(cfg)
        elif cfg.mode == "replay":
            cfg.env_info['has_renderer'] = True
            cfg.env_info["reward_type"] = "dense"
            env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
            env.replay(replay_id = 1, img_post_process_fn=make_imageprocess_fn(cfg), reward_fn=make_reward_fn(cfg, env.action_space.shape), action_pre_process_fn=make_actionprocess_fn(cfg))
        elif cfg.mode == "train":
            env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
            test_env = make_env(cfg.env_name, OmegaConf.to_container(cfg.env_info, resolve=True))
            action_shape = env.action_space.shape
            agent = make_policy_agent(cfg, cfg.agent_name, cfg.device, action_shape, is_train=True)
            agent.train_agent(env, test_env, img_post_process_fn=make_imageprocess_fn(cfg), reward_fn=make_reward_fn(cfg, action_shape), action_pre_process_fn=make_actionprocess_fn(cfg))
        else:
            raise NotImplementedError

if __name__ == "__main__":
    main_policy()