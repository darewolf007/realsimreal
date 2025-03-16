import numpy as np
from simple_sim.environment.base_environment import SingleViewSimulation

class PickApplePlaceBowlSimulation(SingleViewSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.pick_flag = False
        self.place_flag = False
        self.pick_reward_given = False
        self.place_reward_given = False

    def reward(self, info, action):
        if self.env_info['reward_type'] == "sparse":
            reaching_reward = -1
        else:
            raise NotImplementedError
        pick_reward = 0
        place_reward = 0
        self.is_pick_done_from_sim(info, action)
        self.is_place_done_from_sim(info, action)
        self.is_grasp_from_sim(info, action)
        if self.pick_flag and not self.pick_reward_given:
            pick_reward = 100 - self.gripper_change_num * 10
            self.pick_reward_given = True
        if self.place_flag and not self.place_reward_given:
            place_reward = 100 - self.gripper_change_num * 10
            self.place_reward_given = True
            
        return reaching_reward + pick_reward + place_reward
    
    def is_sucess(self, info, action):
        if self.pick_flag and self.place_flag:
            return True
        else:
            return False

    def is_pick_done_from_sim(self, info, action):
        if action[-1] == 1 and (not self.pick_flag) and (self.sub_task_idx==1) and (
            0.10 > info["delta_gripper"] and info["delta_gripper"] > 0.05) and (
                0.08 > info["gripper_apple"]
            ):
            self.pick_flag = True

    def is_place_done_from_sim(self, info, action):
        if action[-1] == -1 and (not self.place_flag) and (self.sub_task_idx==1) and (
            0.15 > info["gripper_bowl"] and info['subtask1'] < 0.11
            ):
            self.place_flag = True
        
    def is_grasp_from_sim(self, info, action):
        if(self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            if info["gripper_apple"] > 0.1:
                self.gripper_change_num += 1

    def reset(self):
        obs = super().reset()
        self.pick_flag = False
        self.place_flag = False
        self.pick_reward_given = False
        self.place_reward_given = False
        return obs