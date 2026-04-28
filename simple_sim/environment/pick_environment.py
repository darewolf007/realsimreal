import numpy as np
from simple_sim.environment.base_environment import SingleViewSimulation

class PickBananaSimulation(SingleViewSimulation):
    def reward(self, info, action):
        if self.env_info['reward_type'] == "sparse":
            reaching_reward = -1
        else:
            reaching_reward = (- info["gripper_banana"]) * 10
        grasp_reward = 0
        if (self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            if info["gripper_banana"] > 0.07:
                self.gripper_change_num += 1
        if self.is_sucess(info, action):
            grasp_reward = 100 - self.gripper_change_num * 10
            reaching_reward = 0
        return reaching_reward + grasp_reward

    def is_sucess(self, info, action):
        if action[-1] == 1 and (
            0.1 > info["delta_gripper"] and info["delta_gripper"] > 0.04) and (
                info["gripper_banana"] < 0.05
            ):
            return True
        else:
            return False
        
class LiftBananaSimulation(SingleViewSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.moving_object = "banana"
        self.target_pose = self.env.sim.data.get_site_xpos(self.moving_object + "_center_site").copy() + np.array([0, 0, 0.05])
        self.pick_reward_given = False
        self.lift_reward_given = False
        self.gripper_change_num = -1 # pick will change gripper once
        self.lift_threshold = 0.05
        if self.env_info['reward_type'] == "dense":
            self.sub_task_reward_scale = 0.25
        else:
            self.sub_task_reward_scale = 5

    def step(self, action, step_num=2):
        self.is_gripper_change(action)
        return super().step(action, step_num)

    def compute_dense_reward(self):
        tcp_pose = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
        moving_object_pose = self.env.sim.data.get_site_xpos(self.moving_object + "_center_site").copy()
        tcp_to_obj_dist = np.linalg.norm(moving_object_pose - tcp_pose)
        reaching_reward = 1 - np.tanh(10.0 * tcp_to_obj_dist)
        is_grasped = self.is_grasping(self.moving_object)
        reward = 0.0
        if is_grasped:
            reward += 1.0 
            reward += 1.0 
            obj_to_goal_dist = np.linalg.norm(self.target_pose - moving_object_pose)
            lift_reward = 1 - np.tanh(5.0 * obj_to_goal_dist)
            reward += lift_reward * 2.0 
            
            # (可选) 额外的 Z 轴高度激励，不仅看距离目标，单纯往上提就有奖
            # 这是一个 dense shaping，帮助打破静止
            # current_height = moving_object_pose[2]
            # reward += current_height * 0.5 
            print(f"Grasped! Lift Reward: {lift_reward:.4f}")
            
        else:
            reward = reaching_reward

        if self.is_lift_done_from_sim():
            reward = 5.0
        reward /= 5.0
        return reward
    
    def reward(self):
        additional_reward = - self.robot_collisions * self.sub_task_reward_scale
        if self.env_info['reward_type'] == "sparse":
            reward = -1
            is_grasped = self.is_grasping(self.moving_object)
            reward += self.is_pick_done_from_sim(is_grasped) * self.sub_task_reward_scale
            reward += self.is_lift_done_from_sim() * self.sub_task_reward_scale
            if self.is_sucess():
                reward = 100- min(self.gripper_change_num, 5) * self.sub_task_reward_scale
        elif self.env_info['reward_type'] == "dense":
            reward = self.compute_dense_reward()
        elif self.env_info['reward_type'] == "online_sparse":
            reward =  -1
        else:
            raise NotImplementedError
        return reward + additional_reward
    
    def is_sucess(self):
        if self.pick_reward_given and self.lift_reward_given:
            return True
        else:
            return False

    def is_pick_done_from_sim(self, is_grasped):
        if is_grasped and not self.pick_reward_given:
            self.pick_reward_given = True
            return True
        else:
            return False

    def is_lift_done_from_sim(self):
        moving_object_pose = self.env.sim.data.get_site_xpos(self.moving_object + "_center_site").copy()
        is_obj_lifted = (np.linalg.norm(self.target_pose - moving_object_pose)<= self.lift_threshold)
        if is_obj_lifted and not self.lift_reward_given:
            self.lift_reward_given = True
            return True
        else:
            return False
        
    def is_gripper_change(self, action):
        if(self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            self.gripper_change_num += 1

    def reset(self):
        obs = super().reset()
        self.gripper_change_num = -1
        self.pick_reward_given = False
        self.lift_reward_given = False
        return obs
    