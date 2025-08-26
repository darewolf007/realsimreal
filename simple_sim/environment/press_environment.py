import numpy as np
from simple_sim.environment.base_environment import SingleViewSimulation

class PressButtonSimulation(SingleViewSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.target_object = "stop_button"
        self.press_reward_given = False
        self.press_threshold = 0.005
        if self.env_info['reward_type'] == "dense":
            self.sub_task_reward_scale = 1
        else:
            self.sub_task_reward_scale = 5

    def reward(self):
        additional_reward = - self.robot_collisions * self.sub_task_reward_scale
        if self.env_info['reward_type'] == "sparse":
            reward = -1
            if self.is_sucess():
                reward = 100
        elif self.env_info['reward_type'] == "dense":
            tcp_pose = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
            target_pose = self.env.sim.data.get_site_xpos(self.target_object + "_center_site").copy() 
            tcp_to_obj_dist = np.linalg.norm(target_pose - tcp_pose)
            reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
            reward = reaching_reward
        elif self.env_info['reward_type'] == "online_sparse":
            reward =  -1
        else:
            raise NotImplementedError
        print("additional_reward", additional_reward)
        return reward + additional_reward

    def is_sucess(self, info, action):
        if action[-1] == 1 and (info["gripper_stop_button"] < 0.001):
            return True
        else:
            return False