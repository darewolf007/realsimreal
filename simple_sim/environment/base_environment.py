import os
import cv2
import torch
import numpy as np
from simple_sim.real_to_simulation import RealInSimulation
from simple_sim.sim_utils.draw_growing_curve import GrowingCurveAnimator

class SingleViewSimulation(RealInSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.gripper_change_num = 0   
        self.reward_collect_list = []

    def step(self, action, step_num=1, use_joint_controller=False):
        observation, _, _, info = super().multi_step(action, self.env_info['use_delta'], use_joint_controller, is_collect=False, step_num=step_num, use_euler= self.env_info['use_euler'])
        done = self.is_sucess()
        reward = self.reward()
        if done:
            info['is_success'] = True
            info['truncation'] = True
            return observation, reward, done, info
        else:
            info['is_success'] = False
        return observation, reward, done, info 

    def reward(self):
        raise NotImplementedError

    def is_sucess(self):
        raise NotImplementedError

    def _reset(self):
        observation = super().reset()
        self.gripper_change_num = 0
        self.reward_collect_list = []
        return observation

    def reset(self):
        observation = self._reset()
        self.init_robot_joint = self.env.robots[0]._joint_positions.copy()
        self.init_robot_eff = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
        self.obj_init_pose = {}
        for lable in self.env_info['obj_info']['labels']:
            self.obj_init_pose[lable] = self.env.sim.data.get_site_xpos(lable + "_center_site").copy()
        return observation
    
    def is_grasping(self, target_object):
        active_obj = self.env.scene_objects[self.env.env_info['obj_info']['labels'].index(target_object)]
        is_grasped = np.array(self.env._check_grasp(gripper=self.env.robots[0].gripper,
                    object_geoms=[g for g in active_obj.contact_geoms])).astype(float)
        return is_grasped

    #TODO remove this part
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

    def replay(self, replay_id = None, img_post_process_fn = None, reward_fn=None, action_pre_process_fn=None, show_img=False):
        pt_data_path = self.env_info['replay_buffer_load_dir']
        chunks = os.listdir(pt_data_path)
        chunks = [c for c in chunks if c[-3:] == ".pt"]
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        path = os.path.join(pt_data_path, chucks[0])
        payload = torch.load(path)
        obses = payload[0]
        actions = np.array(payload[2])
        demo_starts = np.load(os.path.join(pt_data_path, "demo_starts.npy"))
        demo_ends = np.load(os.path.join(pt_data_path, "demo_ends.npy"))
        # actions[:, :3] = actions[:, :3] * 100
        # xyz_mean = np.mean(actions[:, :3], axis=0, keepdims=True)
        # xyz_std = np.std(actions[:, :3], axis=0, keepdims=True)
        # xyz_std[xyz_std == 0] = 1.0
        # actions[:, :3] = (actions[:, :3] - xyz_mean) / xyz_std
        for i in range(len(demo_starts)):
            if replay_id is not None and i != replay_id:
                continue
            self.reset()
            start = demo_starts[i]
            end = demo_ends[i]
            traj_action = actions[start:end]
            demo_obs =obses[start:end]
            for step in range(traj_action.shape[0]):
                step_action = traj_action[step]
                real_action = step_action
                # real_action = action_pre_process_fn(step_action, xyz_mean, xyz_std)
                obs, reward, done, info = self.step(real_action)
                if reward_fn is not None:
                    reward, done, reward_info = reward_fn(obs, real_action, info, reward, reward_type="replay", is_save=False, is_train=False)
                self.reward_collect_list.append(reward)
                if show_img and img_post_process_fn is not None:
                    new_obs = img_post_process_fn(obs)
                    cv2.imshow("replay_obs", np.transpose(new_obs, (1, 2, 0))[:, :, ::-1])
                    cv2.imshow("demo_obs", np.transpose(np.array(demo_obs[step]), (1, 2, 0))[:,:,::-1])
                    cv2.waitKey(1)
                    print(f"test_obs: {new_obs.shape} | test_action: {real_action} | test_reward: {reward} | test_done: {done} | test_reward_info: {reward_info}")
                # for mine
                # time_step = img_post_process_fn(obs, reward = reward, info = info, action = real_action, is_reset=False, is_train = False)
                # additional_reward = 0.98 * (reward_fn.get_reward(time_step.observation[None,:3,...], torch.tensor(1)))
                # self.reward_collect_list.append(reward+additional_reward.cpu().numpy()[0][0])
            print(f"Demo {i} finished.")
            save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../experiments/reward_replay/" + self.env_info['reward_type'] + "/" + self.env_info['task_name'] + "/"
            os.makedirs(save_path, exist_ok=True)
            GrowingCurveAnimator(np.array(self.reward_collect_list), title="Episode Reward").save_gif(os.path.join(save_path, f"reward__{i}.gif"), fps=10, interval_ms=10)
            np.save(os.path.join(save_path, f"reward_collect_list_{i}.npy"), np.array(self.reward_collect_list))
        self.close() # if close, the env can not be used to train agent
        