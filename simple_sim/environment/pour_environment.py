import numpy as np
from simple_sim.environment.base_environment import SingleViewSimulation

class PourCanSimulation(SingleViewSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.grasp_flag = False
        self.pour_flag = False
        self.grasp_reward_given = False
        self.pour_reward_given = False

    def step(self, action, step_num=2):
        return super().step(action, step_num)
    
    def reward(self, info, action):
        if self.env_info['reward_type'] == "sparse":
            reaching_reward = -1
        else:
            raise NotImplementedError
        grasp_reward = 0
        pour_reward = 0
        self.is_grasp_done_from_sim(info, action)
        self.is_pour_done_from_sim(info, action)
        self.is_grasp_from_sim(info, action)
        if self.grasp_flag and not self.grasp_reward_given:
            grasp_reward = 100 - self.gripper_change_num * 10
            self.grasp_reward_given = True
            reaching_reward = 0
        if self.pour_flag and not self.pour_reward_given:
            pour_reward = 100 - self.gripper_change_num * 10
            self.pour_reward_given = True
            reaching_reward = 0
            
        return reaching_reward + grasp_reward + pour_reward
    
    def is_sucess(self, info, action):
        if self.grasp_flag and self.pour_flag:
            return True
        else:
            return False

    def is_grasp_done_from_sim(self, info, action):
        if action[-1] == 1 and (self.sub_task_idx==1) and (
            0.10 > info["delta_gripper"] and info["delta_gripper"] > 0.05) and (
                0.08 > info["gripper_can"]
            ):
            self.grasp_flag = True

    def is_pour_done_from_sim(self, info, action):
        if action[-1] == 1 and self.grasp_flag and (self.sub_task_idx==1) and (
            0.10 > info["delta_gripper"] and info["delta_gripper"] > 0.05) and (
                0.08 > info["gripper_can"] and 0.1 > info["gripper_cup"]
            ):
            self.pour_flag = True
        
    def is_grasp_from_sim(self, info, action):
        if (self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            if info["gripper_can"] < 0.085:
                self.gripper_change_num += 1

    def reset(self):
        obs = super().reset()
        self.grasp_flag = False
        self.pour_flag = False
        self.grasp_reward_given = False
        self.pour_reward_given = False
        return obs