import os
import cv2
import torch
import numpy as np
from simple_sim.real_to_simulation import RealInSimulation

class SingleViewSimulation(RealInSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.gripper_change_num = 0   

    def step(self, action, step_num=1, use_joint_controller=False):
        observation, _, _, info = super().multi_step(action, self.env_info['use_delta'], use_joint_controller, is_collect=False, step_num=step_num, use_euler= self.env_info['use_euler'])
        done = self.is_sucess(info, action)
        reward = self.reward(info, action)
        if done:
            info['is_success'] = True
            return observation, reward, done, info
        else:
            info['is_success'] = False
        return observation, reward, done, info 

    def reward(self, info, action):
        raise NotImplementedError

    def is_sucess(self, info, action):
        raise NotImplementedError

    def reset(self):
        observation = super().reset()
        self.gripper_change_num = 0
        return observation
    
    def action_info(self):
        pt_data_path = self.env_info['replay_buffer_load_dir']
        chunks = os.listdir(pt_data_path)
        chunks = [c for c in chunks if c[-3:] == ".pt"]
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        path = os.path.join(pt_data_path, chucks[0])
        payload = torch.load(path)
        actions = np.array(payload[2])
        actions[:,:3] = actions[:,:3] * 100
        xyz_mean = np.mean(actions[:, :3], axis=0, keepdims=True)
        xyz_std = np.std(actions[:, :3], axis=0, keepdims=True)
        xyz_std[xyz_std == 0] = 1.0
        actions[:, :3] = (actions[:, :3] - xyz_mean) / xyz_std
        return xyz_mean, xyz_std

    def replay(self, img_post_process_fn, reward_fn=None, action_pre_process_fn=None):
        pt_data_path = self.env_info['replay_buffer_load_dir']
        chunks = os.listdir(pt_data_path)
        chunks = [c for c in chunks if c[-3:] == ".pt"]
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        path = os.path.join(pt_data_path, chucks[0])
        payload = torch.load(path)
        obses = payload[0]
        actions = np.array(payload[2])
        actions[:,:3] = actions[:,:3] * 100
        demo_starts = np.load(os.path.join(pt_data_path, "demo_starts.npy"))
        demo_ends = np.load(os.path.join(pt_data_path, "demo_ends.npy"))
        xyz_mean = np.mean(actions[:, :3], axis=0, keepdims=True)
        xyz_std = np.std(actions[:, :3], axis=0, keepdims=True)
        xyz_std[xyz_std == 0] = 1.0
        actions[:, :3] = (actions[:, :3] - xyz_mean) / xyz_std
        self.reset()
        for i in range(len(demo_starts)):
            self.reset()
            start = demo_starts[i]
            end = demo_ends[i]
            traj_action = actions[start:end]
            demo_obs =obses[start:end]
            for step in range(traj_action.shape[0]):
                step_action = traj_action[step]
                real_action = action_pre_process_fn(step_action, xyz_mean, xyz_std)
                obs, reward, done, info = self.step(real_action)
                reward, done, reward_info = reward_fn(obs, real_action, info, reward, reward_type="test", is_save=False, is_train=False)
                # new_obs = img_post_process_fn(obs)
                # print("test_obs", new_obs.shape)
                # print("test_action", real_action)
                # print("test_reward", reward)
                # print("test_done", done)
                # print("test_reward_info", reward_info)
                # cv2.imshow("replay_obs", np.transpose(new_obs, (1, 2, 0))[:,:,::-1])
                # cv2.imshow("demo_obs", np.transpose(np.array(demo_obs[step]), (1, 2, 0))[:,:,::-1])
                # cv2.waitKey(1)
        # self.close() # if close, the env can not be used to train agent
        