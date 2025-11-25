import numpy as np
import transforms3d
from simple_sim.environment.base_environment import SingleViewSimulation

class PourCanSimulation(SingleViewSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.moving_object = "can"
        self.target_object = "cup"
        self.pick_reward_given = False
        self.pour_reward_given = False
        self.gripper_change_num = -1 # pick will change gripper once
        self.pour_threshold = 0.08
        self.lift_threshold = 0.02
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
        pour_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        reward += pour_reward * is_grasped
        print("pour_reward", pour_reward * is_grasped)
        reward += self.is_pour_done_from_sim() * self.sub_task_reward_scale
        print("pour_reward_given", self.pour_reward_given)
        if self.is_sucess():
            reward = 10 - min(self.gripper_change_num, 5) * self.sub_task_reward_scale
        return reward
    
    def compute_dense_reward(self):
        tcp_pose = self.env.sim.data.get_site_xpos('gripper0_right_grip_site').copy()
        moving_object_pose = self.env.sim.data.get_site_xpos(self.moving_object + "_center_site").copy()
        oir_target_pose = self.env.sim.data.get_site_xpos(self.target_object + "_center_site").copy()
        target_pose = oir_target_pose + np.array([0, 0, 0.07])
        end_to_obj = np.linalg.norm(tcp_pose - moving_object_pose)
        is_contact = end_to_obj < 0.065
        all_contact = self.is_grasping(self.moving_object)
        reward = -0.1 * end_to_obj
        dump_vector_xy = target_pose[:2] - moving_object_pose[:2]
        sign_x = 1 if dump_vector_xy[0] < -1e-6 else -1
        sign_y = 1 if dump_vector_xy[1] < -1e-6 else -1
        print("init", self.init_robot_joint, self.init_robot_eff, self.obj_init_pose)
        if is_contact:
            reward += 0.1
            if all_contact:
                reward += 0.2
            lift = np.linalg.norm(self.obj_init_pose[self.moving_object][2] - moving_object_pose[2])
            reward += 50 * lift
            condition = lift > 0.03
            if condition and all_contact:  # if object off the table
                obj_target_distance = np.linalg.norm(moving_object_pose - target_pose)
                reward += 2.0  # bonus for lifting the object
                reward += -0.5 * np.linalg.norm(tcp_pose - target_pose)  # make hand go to target
                reward += -1.5 * obj_target_distance  # make object go to target
                print("reward", reward)
                if obj_target_distance < 0.09:
                    reward += 1 / (max(obj_target_distance, 0.03))
                    obj_quat = self.env.sim.data.get_body_xquat(self.moving_object+"_main").copy()
                    z_axis = transforms3d.quaternions.quat2mat(obj_quat) @ np.array([0, 0, 1])
                    reward += -(sign_x * z_axis[0] + sign_y * z_axis[1]) * 20 - abs(z_axis[0] - z_axis[1]) * 10
                    if (sign_x * z_axis[0] < 0) and (sign_y * z_axis[1] < 0):
                        reward += np.arccos(z_axis[2]) * 100
        if self.robot_collisions:
            reward -= 0.05 * abs(reward)
        max_reward = 0.3 + 50 * target_pose[2] + 2.0 + 1 / 0.03 + 20 * 1.4142 + 100
        reward /= max_reward
        return reward
    
    def reward(self):
        additional_reward = - self.robot_collisions * self.sub_task_reward_scale
        if self.env_info['reward_type'] == "sparse":
            reward = -1
            is_grasped = self.is_grasping(self.moving_object)
            reward += self.is_pick_done_from_sim(is_grasped) * self.sub_task_reward_scale
            reward += self.is_pour_done_from_sim() * self.sub_task_reward_scale
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
        if self.pick_reward_given and self.pour_reward_given:
            return True
        else:
            return False

    def is_pick_done_from_sim(self, is_grasped):
        if is_grasped and not self.pick_reward_given:
            self.pick_reward_given = True
            return True
        else:
            return False

    def is_pour_done_from_sim(self):
        moving_object_pose = self.env.sim.data.get_site_xpos(self.moving_object + "_center_site").copy()
        target_pose = self.env.sim.data.get_site_xpos(self.target_object + "_center_site").copy() 
        all_contact = self.is_grasping(self.moving_object)
        dump_vector_xy = target_pose[:2] - moving_object_pose[:2]
        sign_x = 1 if dump_vector_xy[0] < -1e-6 else -1
        sign_y = 1 if dump_vector_xy[1] < -1e-6 else -1
        obj_target_distance = np.linalg.norm(moving_object_pose - target_pose)
        if all_contact and obj_target_distance < 0.05:
            obj_quat = self.env.sim.data.get_body_xquat(self.moving_object+"_main").copy()
            z_axis = transforms3d.quaternions.quat2mat(obj_quat) @ np.array([0, 0, 1])
            if (sign_x * z_axis[0] < 0) and (sign_y * z_axis[1] < 0):
                reward = np.arccos(z_axis[2]) * 100
                if reward > 108 and not self.pour_reward_given:
                    self.pour_reward_given = True
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
        self.pour_reward_given = False
        return obs