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
    