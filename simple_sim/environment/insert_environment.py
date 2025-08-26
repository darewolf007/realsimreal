import numpy as np
from simple_sim.environment.base_environment import SingleViewSimulation

class InsertMarkerSimulation(SingleViewSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.moving_object = "marker"
        self.target_object = "pen_holder"
        self.pick_reward_given = False
        self.insert_reward_given = False
        self.gripper_change_num = -2 # pick insert will change gripper once
        self.insert_threshold = 0.05
        if self.env_info['reward_type'] == "dense":
            self.sub_task_reward_scale = 1
        else:
            self.sub_task_reward_scale = 5

    def step(self, action, step_num=2):
        self.is_gripper_change(action)
        return super().step(action, step_num)

    def compute_dense_reward(self):
        tcp_pose = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
        moving_object_pose = self.env.sim.data.get_site_xpos(self.moving_object + "_center_site").copy()
        target_pose = self.env.sim.data.get_site_xpos(self.target_object + "_center_site").copy() 
        tcp_to_obj_dist = np.linalg.norm(moving_object_pose - tcp_pose)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward
        print("reaching_reward", reaching_reward)
        is_grasped = self.is_grasping(self.moving_object)
        if is_grasped:
            reward += self.is_pick_done_from_sim(is_grasped) * self.sub_task_reward_scale
        print("is_grasped", is_grasped)
        print("pick_reward_given", self.pick_reward_given)
        obj_to_goal_dist = np.linalg.norm(target_pose - moving_object_pose)
        insert_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        reward += insert_reward * is_grasped
        print("insert_reward", insert_reward * is_grasped)
        reward += self.is_insert_done_from_sim() * self.sub_task_reward_scale
        print("insert_reward_given", self.insert_reward_given)
        if self.is_sucess():
            reward = 10 - min(self.gripper_change_num, 5) * self.sub_task_reward_scale
        return reward
    
    
    def reward(self):
        additional_reward = - self.robot_collisions * self.sub_task_reward_scale
        if self.env_info['reward_type'] == "sparse":
            reward = -1
            is_grasped = self.is_grasping(self.moving_object)
            reward += self.is_pick_done_from_sim(is_grasped) * self.sub_task_reward_scale
            reward += self.is_insert_done_from_sim() * self.sub_task_reward_scale
            if self.is_sucess():
                reward = 100- min(self.gripper_change_num, 5) * self.sub_task_reward_scale
        elif self.env_info['reward_type'] == "dense":
            reward = self.compute_dense_reward()
        elif self.env_info['reward_type'] == "online_sparse":
            reward =  -1
        else:
            raise NotImplementedError
        print("additional_reward", additional_reward)
        return reward + additional_reward
    
    def is_sucess(self):
        if self.pick_reward_given and self.insert_reward_given:
            return True
        else:
            return False

    def is_pick_done_from_sim(self, is_grasped):
        if is_grasped and not self.pick_reward_given:
            self.pick_reward_given = True
            return True
        else:
            return False

    def is_insert_done_from_sim(self):
        moving_object_pose = self.env.sim.data.get_site_xpos(self.moving_object + "_center_site").copy()
        target_pose = self.env.sim.data.get_site_xpos(self.target_object + "_center_site").copy() 
        is_obj_inserted = (np.linalg.norm(target_pose - moving_object_pose)<= self.insert_threshold)
        if is_obj_inserted and not self.insert_reward_given:
            self.insert_reward_given = True
            return True
        else:
            return False
        
    def is_gripper_change(self, action):
        if(self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            self.gripper_change_num += 1

    def reset(self):
        obs = super().reset()
        self.gripper_change_num = -2
        self.pick_reward_given = False
        self.insert_reward_given = False
        return obs