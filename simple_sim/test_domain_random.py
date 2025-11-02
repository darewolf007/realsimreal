import mujoco
import os
import numpy as np
import real_to_simulation
import robosuite.macros as macros
from robosuite.utils.input_utils import *
from simple_sim.sim_utils.domain_randomization_wrapper import DomainRandomizationWrapper
from simple_sim.sim_utils.sim_util import transform_to_camera_frame, matrix_to_translation_quaternion, get_handeye_T

# We'll use instance randomization so that entire geom groups are randomized together
macros.USING_INSTANCE_RANDOMIZATION = True

if __name__ == "__main__":
    task_name = "Pour can into a cup"
    subtask_1 = "Pick up the can"
    subtask_2 = "Pour the can into the cup"
    subtask_1_obj = ["gripper", "can"]
    subtask_2_obj = ["can", "cup"]
    base_path = os.path.dirname(os.path.realpath(__file__))
    handeye_T_path = os.path.join(base_path, "../configs/robot/ur5_kinect_handeyecalibration_eye_on_base.yaml")
    handeye_T = get_handeye_T(handeye_T_path)
    robot_init_pose = np.array([ -1.30487138, -1.69159379, 1.7358554 , -1.55820926, -1.51700765,
       -0.55815155])
    can_pose = np.array([-2.58006106e-01,  4.77104923e-01,  0.04,  0.707, 0, 0,  0.707])
    cup_pose = np.array([-0.42696884,  0.23760321,  0.025,  1, 0, 0,  0])
    scene_dict = {"labels": ["can", "cup"], "poses": [can_pose, cup_pose], "grasp_obj": [True, False]}
    replay_data_save_path = os.path.join(base_path, "../data/sim_data/" + task_name.replace(" ", "_") + "/")
    env_info = {}
    env_info['obj_pose_base'] = "robot"
    env_info['replay_data_save_path'] = replay_data_save_path
    env_info['task_name'] = task_name
    env_info['subtask_language_info'] = [subtask_1, subtask_2]
    env_info['subtask_object_info'] = [subtask_1_obj, subtask_2_obj]
    env_info['hand_eye_path'] = handeye_T_path
    env_info['hand_eye'] = handeye_T
    env_info['obj_info'] = scene_dict
    env_info['use_gravity'] = True
    env_info['data_path'] = "/home/haowen/hw_mine/Real_Sim_Real/data/real_data/easy_task/pour_can/5/traj/"
    begin_step = 3
    # env_info['base_choose'] = "camera"
    env_info['base_choose'] = "robot"
    robot_init_pose = np.array([ 1.85383064, -1.74503436, -1.01362259, -1.64450421, -1.57473976, -0.25406391])
    robot_init_pose = np.load(env_info['data_path'] + "joint_" + str(begin_step-1) + ".npy")
    robot_init_pose[0], robot_init_pose[2] = robot_init_pose[2], robot_init_pose[0]
    env_info['robot_init_qpos'] = robot_init_pose
    env_info['camera_depths'] = True
    env_info['is_crop'] = False
    env_info['crop_image_size'] = (768, 768)
    if env_info['is_crop']:
        env_info['camera_heights'] = [768*2, 1536, 1536, 1536]
        env_info['camera_widths'] = [2048, 2048, 2048, 2048]
    else:
        env_info['camera_heights'] = [768*2, 1536, 1536, 1536, 1536, 1536]
        env_info['camera_widths'] = [2048, 2048, 2048, 2048, 2048, 2048]
    env_info['camera_names'] = ["sceneview", "birdview", "frontview", "rightview", "robot0_eye_in_hand", "moveview"]
    env_info['has_renderer'] = True
    env_info['control_freq'] = 20
    env_info['task_max_step'] = 200
    env_info['subtask_max_step'] = 50
    env_info['use_euler'] = True
    env_info['action_noise'] = False
    env_info['init_noise'] = False
    env_info['use_delta'] = True
    env_info['init_translation_noise_bounds'] = (-0.03, 0.03)
    env_info['init_rotation_noise_bounds'] = (-50, 50)
    env_info['use_joint_controller'] = False
    env_info['max_action'] = 4
    test_real = real_to_simulation.RealInSimulation("UR5e",
                                 env_info,
                                 has_renderer=True,
                                 has_offscreen_renderer=True,
                                 render_camera="sceneview",
                                 ignore_done=True,
                                 use_camera_obs=True,
                                 camera_depths=env_info['camera_depths'],
                                 control_freq=env_info['control_freq'],
                                 renderer="mjviewer",
                                 camera_heights=env_info['camera_heights'],
                                 camera_widths=env_info['camera_widths'],
                                 camera_names=env_info['camera_names'],
                                 schedule_random = True)
    test_real.reset()
    # env = DomainRandomizationWrapper(testenv.env)
    # env.reset()
    # env.viewer.set_camera(camera_id=0)

    # Get action limits
    # low, high = env.action_spec

    # do visualization
    while(True):
        test_real.reset()
        for _ in range(10):
            observations,_,_,_ = test_real.multi_step(np.array([0, 0, 0, 0, 0, 0, 1]))
        #     import cv2
        #     cv2.imshow("sceneview", observations['sceneview_image'])
        #     cv2.waitKey(2)
        # cv2.destroyAllWindows()
        # import matplotlib.pyplot as plt
        # plt.imshow(observations['crop_sceneview_image'])
        # plt.show()
