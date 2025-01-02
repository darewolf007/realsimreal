import time
import cv2
import numpy as np
from base_env import SimpleEnv
from robotic_ik import mink_ik
from scipy.spatial.transform import Rotation as R
from sim_utils import transform_to_camera_frame, matrix_to_translation_quaternion

class RealInSimulation:
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        if env_info['base_choose'] == "camera":
            translation, quaternion = matrix_to_translation_quaternion(np.linalg.inv(env_info['hand_eye']))
            rotation_z_180 = np.array([0, 0, 1, 0])
            rotation = R.from_quat(quaternion) * R.from_quat(rotation_z_180)
            new_quaternion = rotation.as_quat()[[3, 0, 1, 2]]
            env_info['hand_eye'] = np.concatenate([translation, new_quaternion])
        self.env = SimpleEnv(robot, env_info, has_renderer = has_renderer, *args, **kwargs)
        self.env_info = env_info
        self.has_renderer = has_renderer
        self.init_invese_kinematics()
        self.reset()
        
    def init_invese_kinematics(self):
        mujoco_model = self.env.get_mujoco_model()
        self.mink_ik = mink_ik(mujoco_model)

    def step(self, action, use_joint_controller=False):
        if not use_joint_controller:
            translation = action[:3]
            quaternion = action[3:7]
            qpos = self.mink_ik.ik(self.env.sim.data, translation, quaternion, self.env.model_timestep)[:self.env.robots[0].dof]
        gripper_data = action[-1]
        action = np.concatenate([qpos, np.array([gripper_data])])
        print("action", action)
        observations, reward, done, info = self.env.step(action)

        rightview_image = observations['rightview_image']
        rightview_image = cv2.flip(rightview_image, 0)
        corrected_image = cv2.cvtColor(rightview_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('right_view_image.png', corrected_image)

        front_view_image = observations['frontview_image']
        front_view_image = cv2.flip(front_view_image, 0)
        corrected_image = cv2.cvtColor(front_view_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('front_view_image.png', corrected_image)

        bird_view_image = observations['birdview_image']
        bird_view_image = cv2.flip(bird_view_image, 0)
        corrected_image = cv2.cvtColor(bird_view_image, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite('bird_view_image.png', corrected_image)
        # front_view_image = cv2.flip(front_view_image, 1)
        # import matplotlib.pyplot as plt
        # plt.imshow(front_view_image)
        # plt.show()
        # front_view_image = cv2.filp(front_view_image, 0)
        # cv2.imwrite('front_view_image.png', front_view_image)
        # time.sleep(10)
        #     print("now", env.sim.data.get_site_xpos('robot0_attachment_site'))
        #     print("gripper1", env.sim.data.get_site_xpos('gripper0_right_ft_frame'))
        #     print("gripper2", env.sim.data.get_site_xpos('gripper0_right_grip_site_cylinder'))
        return observations, reward, done, info

    def replay_demonstration(self, is_collect = False):
        print(self.env.sim.model.body_pos[21])
        print(self.env.sim.model.body_quat[21])
        for i in range(1, 53):
            collect_data = np.load(self.env_info['data_path'] + "traj_" + str(i) + ".npy")
            translation_data = collect_data[:3] + np.array([0, 0, 0.2])
            rotation_data = collect_data[3:-1]
            if collect_data[-1] == 0:
                gripper_data = -1
            else:
                gripper_data = 1
            if self.env_info['base_choose'] == "camera":
                translation, xyzw_quaternion = transform_to_camera_frame(handeye_T, rotation_data, translation_data)
                quaternion = xyzw_quaternion[[3, 0, 1, 2]]
            else:
                raise NotImplementedError
            action = np.concatenate([translation, quaternion, np.array([gripper_data])])
            self.step(action)
            if self.has_renderer:
                self.env.render()
            break    
    def reset(self):
        self.env.reset()
        if not self.env_info['use_gravity']:
            self.env.sim.model.opt.gravity[:] = [0.0, 0.0, 0.0]

if __name__ == "__main__":
    handeye_T = np.array(
        [[-0.71984259, -0.2949338,   0.6283635,  -0.76957957],
        [-0.69408555,  0.29477574, -0.65677432,  0.82485058],
        [ 0.00847863, -0.90891216, -0.41690143,  0.23738398],
        [ 0.,          0. ,         0.,          1.  ,      ]]
        )
    can_pose = np.array([-0.06795219, 0.11318543, 0.44714868, -0.23037968, -0.26558182, 0.7429639, -0.56955785])
    scene_dict = {"labels": ["can"], "poses": [can_pose]}
    env_info = {}
    env_info['hand_eye'] = handeye_T
    env_info['obj_info'] = scene_dict
    env_info['use_gravity'] = False
    env_info['data_path'] = "/home/haowen/hw_mine/Real_Sim_Real/data/4/traj/"
    env_info['base_choose'] = "camera"
    env_info['robot_init_qpos'] = np.array([-1.169, -1.19, 1.332, -1.699, -1.518, -0.52])
    env_info['max_reward'] = 1

    test_real = RealInSimulation("UR5e",
                                 env_info,
                                 has_renderer=True,  # no on-screen renderer
                                 has_offscreen_renderer=True,  # no off-screen renderer
                                 ignore_done=True,
                                 use_camera_obs=True,  # no camera observations
                                 control_freq=20,
                                 renderer="mjviewer",
                                 camera_heights=[1536, 1536, 1536],
                                 camera_widths=[2048, 2048, 2048],
                                 camera_names=["birdview", "frontview", "rightview"],)
    test_real.reset()
    test_real.replay_demonstration()