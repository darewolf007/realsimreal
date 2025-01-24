import os
import time
import cv2
import pickle
import torch
import argparse
import numpy as np
import gymnasium as gym
from utils.data_convert_pt import convert_pickles_to_pt
from utils.image_util import resize_image, save_image_pkl
from simple_sim.real_to_simulation import RealInSimulation
from reward_model.online_reward_model import ask_grasp_subtask, ask_pour_subtask
from agent_policy.few_shot_RL.policy import FewDemoPolicy
from agent_policy.few_shot_RL.bc_sac_policy import BCSACPolicy
from pre_train.s3d.s3dg import S3D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="Pick up banana20")
    parser.add_argument("--dataset_name", default="pick up banana")
    parser.add_argument("--task_max_step", type=int, default= 40)
    parser.add_argument("--add_additional_reward", default= False, action="store_true")
    parser.add_argument("--add_bc", default= True, action="store_true")
    parser.add_argument("--gpu_id", default= "0")
    parser.add_argument("--is_crop", default=False, action="store_true")
    parser.add_argument("--train_subtask", default=True, action="store_true")
    parser.add_argument("--crop_image_size", default=(768, 768))
    parser.add_argument("--camera_heights", type=int, nargs='+', default=[1536, 1536, 1536, 1536])
    parser.add_argument("--camera_widths", type=int, nargs='+', default=[2048, 2048, 2048, 2048])
    args = parser.parse_args()
    return args

class PickBananaSimulation(RealInSimulation):
    def __init__(self, robot, env_info, has_renderer, *args, **kwargs):
        super().__init__(robot, env_info, has_renderer, *args, **kwargs)
        self.grasp_flag = False
        self.task_data_path = "./experiments/Pour_can_into_cup/promot_data/all"
        if env_info['use_joint_controller']:
            self.action_space = gym.spaces.Box(low=-env_info['max_action'], high=env_info['max_action'], shape=(self.env.robots[0].dof + 1,), dtype=float)
        else:
            if env_info['use_euler']:
                self.action_space = gym.spaces.Box(low=-env_info['max_action'], high=env_info['max_action'], shape=(7,), dtype=float)
            else:
                self.action_space = gym.spaces.Box(low=-env_info['max_action'], high=env_info['max_action'], shape=(8,), dtype=float)
        self.ask_grasp = False
        self.last_frame = []
        self.grasp_demo_embedding = []
        self.done_demo_embedding = []
        self.last_info = None
        if self.env_info['add_additional_reward']:
            self.init_vido_embedding()
            self.pre_process_pt()
     
    def pre_process_pt(self):
        pt_data_path = self.env_info['replay_buffer_load_dir']
        chunks = os.listdir(pt_data_path)
        chunks = [c for c in chunks if c[-3:] == ".pt"]
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        path = os.path.join(pt_data_path, chucks[0])
        payload = torch.load(path)
        obses = payload[0]
        next_obses = payload[1]
        actions = payload[2]
        self.demo_starts = np.load(os.path.join(pt_data_path, "demo_starts.npy"))
        self.demo_ends = np.load(os.path.join(pt_data_path, "demo_ends.npy"))
        for i in range(len(self.demo_starts)):
            start = self.demo_starts[i]
            end = self.demo_ends[i]
            if (start - end) % 2 != 0:
                end -= 1
            frames = next_obses[start:end]
            frames = frames[None, :,:,:,:]
            video = frames.permute(0, 2, 1, 3, 4)
            video_output = self.net(video.float())
            target_embedding = video_output['video_embedding']
            self.done_demo_embedding.append(target_embedding)
        for i in range(len(self.demo_starts)):
            start = self.demo_starts[i]
            end = self.demo_ends[i]
            action = actions[start:end, -1]
            transition_indices = np.where((action[:-1] == -1) & (action[1:] == 1))[0][0]
            if transition_indices % 2 != 0:
                transition_indices -= 1
            frames = next_obses[start:start+transition_indices]
            frames = frames[None, :,:,:,:]
            video = frames.permute(0, 2, 1, 3, 4)
            video_output = self.net(video.float())
            target_embedding = video_output['video_embedding']
            self.grasp_demo_embedding.append(target_embedding)

    def init_vido_embedding(self):
        self.net = S3D('./pre_train/s3d/s3d_dict.npy', 512)
        self.net.load_state_dict(torch.load('./pre_train/s3d/s3d_howto100m.pth'))
        self.net = self.net.eval()

    def step(self, action, use_delta=True, use_joint_controller=False):
        action[:3] = np.clip(action[:3], -self.env_info['max_action'], self.env_info['max_action'])
        action[:3] = action[:3]/100
        action[-1] = 1 if action[-1] > 0 else -1
        # reward = self.update_reward_online(action)
        self.is_grasp_from_sim(self.last_info, action)
        next_observation, _, _, info = super().multi_step(action, self.env_info['use_delta'], use_joint_controller, is_collect=True, step_num=1, use_euler= self.env_info['use_euler'])
        self.last_info = info
        if self.env_info['is_crop']:
            obs = resize_image(next_observation['crop_sceneview_image'], 1/6)
        else:
            obs = resize_image(next_observation['sceneview_image'], 1/12)
        obs = np.transpose(obs, (2, 0, 1))
        self.last_frame.append(obs)
        if self.env_info["train_subtask"]:
            done = self.is_grasp_done_from_sim(info, action)
        info['is_success'] = False
        reward = self.update_reward(info, action)
        # if self.robot_collisions or info['robot_limits']:
        #     print("collision")
        #     return obs, -10, True, info
        if info["truncation"] == True:
            if self.env_info['add_additional_reward']:
                reward += self.update_done_additional_reward()
        if done:
            info['is_success'] = True
            return obs, reward, True, info
        return obs, reward, False, info
    
    def update_reward_online(self, action):
        if not self.ask_grasp and not self.grasp_flag and (self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            print("in update reward")
            self.grasp_flag = self.is_grasp(self.last_observation)
            self.ask_grasp = True
            # if self.grasp_flag:
            #     return 100
        return -1

    def update_done(self, next_observation):
        if self.grasp_flag and (self.sub_task_idx==1):
                return True
        # if self.grasp_flag and (self.all_task_max_num - self.all_task_step_num < 30):
        #     if self.all_task_step_num % 10 == 0:
        #         print("in update done")
        #         done = self.is_done(next_observation)
        #         return done
        return False

    def is_done(self, observations):
        moving_obj = self.env_info['subtask_object_info'][self.sub_task_idx][0]
        target_obj = self.env_info['subtask_object_info'][self.sub_task_idx][1]
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.4),
            "right_view": resize_image(observations["rightview_image"], 0.4),
            "bird_view": resize_image(observations["birdview_image"], 0.4)
        }
        done_flag = ask_pour_subtask(image_dict, moving_obj, target_obj)
        if self.env_info['save_online_image']:
            image_dict['result'] = done_flag
            save_image_pkl(image_dict, self.env_info['online_data_save_path'] + "/done/", True)
        return done_flag

    def is_grasp(self, observations):
        moving_obj = self.env_info['subtask_object_info'][self.sub_task_idx][0]
        target_obj = self.env_info['subtask_object_info'][self.sub_task_idx][1]
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.25),
            "right_view": resize_image(observations["rightview_image"], 0.25),
            "bird_view": resize_image(observations["birdview_image"], 0.25)
        }
        grasp_flag = ask_grasp_subtask(image_dict, moving_obj, target_obj)
        if self.env_info['save_online_image']:
            image_dict['result'] = grasp_flag
            save_image_pkl(image_dict, self.env_info['online_data_save_path'] + "/grasp/", True)
        return grasp_flag

    def is_pour(self):
        pass

    def update_grasp_additional_reward(self):
        if len(self.last_frame)  % 2 != 0:
            last_frames = np.array(self.last_frame[:-1])
        else:
            last_frames = np.array(self.last_frame)
        if last_frames.ndim != 4:
            return -1
        grasp_video = torch.from_numpy(last_frames)
        grasp_video = grasp_video[None, :,:,:,:]
        grasp_video = grasp_video.permute(0, 2, 1, 3, 4)
        grasp_video_output = self.net(grasp_video.float())
        last_grasp_embedding = grasp_video_output['video_embedding']
        max_reward = -float('inf')
        for demo_embedding in self.grasp_demo_embedding:
            similarity_matrix = torch.matmul(demo_embedding, last_grasp_embedding.t())
            demo_reward = similarity_matrix.detach().numpy()[0][0]
            if demo_reward > max_reward:
                max_reward = demo_reward
        return max_reward
    
    def update_done_additional_reward(self):
        if len(self.last_frame)  % 2 != 0:
            last_frames = np.array(self.last_frame[:-1])
        else:
            last_frames = np.array(self.last_frame)
        grasp_video = torch.from_numpy(last_frames)
        grasp_video = grasp_video[None, :,:,:,:]
        grasp_video = grasp_video.permute(0, 2, 1, 3, 4)
        grasp_video_output = self.net(grasp_video.float())
        last_grasp_embedding = grasp_video_output['video_embedding']
        max_reward = -float('inf')
        for demo_embedding in self.done_demo_embedding:
            similarity_matrix = torch.matmul(demo_embedding, last_grasp_embedding.t())
            demo_reward = similarity_matrix.detach().numpy()[0][0]
            if demo_reward > max_reward:
                max_reward = demo_reward
        return max_reward

    def update_reward(self, info, action):
        dist = info["subtask" + str(self.sub_task_idx)]
        if self.last_dist is None:
            self.last_dist = dist
            return 0
        else:
            delta_dist = self.last_dist - dist
        self.last_dist = dist
        reaching_reward = delta_dist * 10 - 0.5 * dist
        if info["gripper_banana"] < 0.06:
            reaching_reward += 2
        grasp_reward = 0
        if self.is_grasp_done_from_sim(info, action):
            grasp_reward = 10 
        return reaching_reward + grasp_reward

    def is_grasp_from_sim(self, info, action):
        if not self.ask_grasp and not self.grasp_flag and (self.last_action is not None and self.last_action[-1] == -1 and action[-1] == 1):
            if info["gripper_banana"] < 0.05:
                self.grasp_flag = True
            self.ask_grasp = True

    def is_grasp_done_from_sim(self, info, action):
        if action[-1] == 1 and (
            0.1 > info["delta_gripper"] and info["delta_gripper"] > 0.04) and (
                info["gripper_banana"] < 0.05
            ):
            return True
        else:
            return False

    def reset(self):
        self.last_info = None
        self.last_dist = None
        self.grasp_flag = False
        self.ask_grasp = False
        self.last_frame = []
        observation = super().reset()
        if self.env_info['is_crop']:
            obs = resize_image(observation['crop_sceneview_image'], 1/6)
        else:
            obs = resize_image(observation['sceneview_image'], 1/12)
        obs = np.transpose(obs, (2, 0, 1))
        return obs

def set_params(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    task_name = args.task_name
    subtask_1 = "Pick up banana"
    subtask_1_obj = ["gripper", "banana"]
    base_path = os.path.dirname(os.path.realpath(__file__))
    work_dir = os.path.join(base_path, "./experiments/" + task_name)
    real_data_dir = os.path.join(base_path, "./data/sim_data/" + task_name)
    handeye_T_path = os.path.join(base_path, "./configs/ur5_kinect_handeyecalibration_eye_on_base.yaml")
    replay_data_save_path = os.path.join(base_path, "./data/sim_data/pt_data/")
    robot_init_pose = np.array([ 1.85383064, -1.74503436, -1.01362259, -1.64450421, -1.57473976, -0.25406391])
    robot_init_pose[0], robot_init_pose[2] = robot_init_pose[2], robot_init_pose[0]
    env_info = {}
    # env_info['obj_pose_base'] = "camera"
    # can_pose = np.array([[-0.29022616147994995, 0.9859233784675598, -0.04448934271931648, -0.21637749671936035], [-0.5486288070678711, -0.12811657786369324, 0.8261916637420654, 0.23622964322566986], [0.7840760350227356, 0.26419055461883545, 0.5616299510002136, 0.5847076177597046], [0.0, 0.0, 0.0, 1.0]])
    # cup_pose = np.array([[0.43723738193511963, 0.8989970684051514, -0.02505527436733246, 0.09150402992963791], [0.29450204968452454, -0.16944658756256104, -0.9405087232589722, 0.18733008205890656], [-0.8497599959373474, 0.4038466811180115, -0.3388448655605316, 0.6819711923599243], [0.0, 0.0, 0.0, 1.0]])
    env_info['obj_pose_base'] = "robot"
    banana_pose = np.array([-0.29156371,  0.37589715,  0.04,  0.707, 0, 0,  0.707])
    scene_dict = {"labels": ["banana"], "poses": [banana_pose], "grasp_obj": [True]}
    env_info['online_data_save_path'] = os.path.join(work_dir, "promot_data")
    env_info['replay_data_save_path'] = replay_data_save_path
    env_info['task_name'] = task_name
    env_info['subtask_language_info'] = [subtask_1]
    env_info['subtask_object_info'] = [subtask_1_obj]
    env_info['hand_eye_path'] = handeye_T_path
    env_info['obj_info'] = scene_dict
    env_info['use_gravity'] = True
    env_info['is_crop'] = args.is_crop
    env_info['data_path'] = None
    env_info['save_online_image'] = True
    # env_info['base_choose'] = "camera"
    env_info['base_choose'] = "robot"
    env_info['robot_init_qpos'] = robot_init_pose
    env_info['camera_depths'] = True
    env_info['crop_image_size'] = args.crop_image_size
    env_info['camera_heights'] = args.camera_heights
    env_info['camera_widths'] = args.camera_widths
    env_info['camera_names'] = ["sceneview", "birdview", "frontview", "rightview"]
    env_info['has_renderer'] = False
    env_info['control_freq'] = 20
    env_info['use_joint_controller'] = False
    env_info['max_action'] = 4
    env_info['init_noise'] = True
    env_info['init_translation_noise_bounds'] = (-0.005, 0.005)
    env_info['init_rotation_noise_bounds'] = (-5, 5)
    env_info['task_max_step'] = args.task_max_step
    env_info['subtask_max_step'] = 50
    env_info['use_euler'] = True
    env_info['use_delta'] = True
    env_info['add_additional_reward'] = args.add_additional_reward
    env_info['train_subtask'] = args.train_subtask
    if env_info['is_crop']:
        env_info['camera_heights'] = [768*2, 1536, 1536, 1536]
        env_info['camera_widths'] = [2048, 2048, 2048, 2048]
    else:
        env_info['camera_heights'] = [768*2, 1536, 1536, 1536]
        env_info['camera_widths'] = [2048, 2048, 2048, 2048]
    replay_buffer_load_dir = replay_data_save_path + args.dataset_name
    env_info['replay_buffer_load_dir'] = replay_buffer_load_dir

    policy_params = {
    "work_dir": work_dir,
    "task_name": task_name,
    "sub_task_promot": [subtask_1],
    "replay_buffer_capacity": 300000,
    "replay_buffer_load_dir": replay_buffer_load_dir,
    "replay_buffer_keep_loaded": True,
    "pretrain_mode": None,
    "pre_transform_image_size": 128,
    "image_size": 112,
    "cameras": [0],
    "observation_type": "pixel",
    "reward_type": "sparse",
    "control": None,
    "action_repeat": 1,
    "frame_stack": 1,
    "num_updates": 1,
    "model_dir": None,
    "model_step": 40000,
    "agent_name": "rad_sac",
    "init_steps": 2000,
    "num_train_steps": 250000,
    "bc_train_steps": 20,
    "batch_size": 128,
    "hidden_dim": 1024,
    "eval_freq": 1000,
    "num_eval_episodes": 2,
    "critic_lr": 0.001,
    "critic_beta": 0.9,
    "critic_tau": 0.01,
    "critic_target_update_freq": 2,
    "actor_lr": 0.001,
    "actor_beta": 0.9,
    "actor_log_std_min": -10,
    "actor_log_std_max": 2,
    "actor_update_freq": 2,
    "encoder_type": "pixel",
    "encoder_feature_dim": 32,
    "encoder_tau": 0.05,
    "num_layers": 4,
    "num_filters": 32,
    "latent_dim": 128,
    "discount": 0.99,
    "init_temperature": 0.1,
    "alpha_lr": 1e-4,
    "alpha_beta": 0.5,
    "seed": 1,
    "save_tb": True,
    "save_buffer": True,
    "save_video": True,
    "save_sac": True,
    "detach_encoder": False,
    "v_clip_low": -100,
    "v_clip_high": 100,
    "action_noise": None,
    "final_demo_density": 0.5,
    "data_augs": "center_crop",
    "log_interval": 200,
    "conv_layer_norm": True,
    "p_reward": 1,
    }
    return env_info, policy_params

def trainer(args):
    env_info, policy_params = set_params(args)
    env = PickBananaSimulation("UR5e",
                        env_info,
                        has_renderer=env_info['has_renderer'],
                        has_offscreen_renderer=True,
                        render_camera=env_info['camera_names'][0],
                        ignore_done=True,
                        use_camera_obs=True,
                        camera_depths=env_info['camera_depths'],
                        control_freq=env_info['control_freq'],
                        renderer="mjviewer",
                        camera_heights=env_info['camera_heights'],
                        camera_widths=env_info['camera_widths'],
                        camera_names=env_info['camera_names'],)
    if args.add_bc:
        policy = BCSACPolicy(env, torch.device("cuda"), policy_params)
    else:
        policy = FewDemoPolicy(env, torch.device("cuda"), policy_params)
    policy.train()

if __name__ == "__main__":
    args = parse_args()
    # trainer(args)
    env_info, policy_params = set_params(args)
    env = PickBananaSimulation("UR5e",
                        env_info,
                        has_renderer=True,
                        has_offscreen_renderer=True,
                        render_camera=env_info['camera_names'][0],
                        ignore_done=True,
                        use_camera_obs=True,
                        camera_depths=env_info['camera_depths'],
                        control_freq=env_info['control_freq'],
                        renderer="mjviewer",
                        camera_heights=env_info['camera_heights'],
                        camera_widths=env_info['camera_widths'],
                        camera_names=env_info['camera_names'],)
    
    pt_data_path = env_info['replay_buffer_load_dir']
    chunks = os.listdir(pt_data_path)
    chunks = [c for c in chunks if c[-3:] == ".pt"]
    chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
    path = os.path.join(pt_data_path, chucks[0])
    payload = torch.load(path)
    obses = payload[0]
    actions = payload[2]
    actions[:,:3] = actions[:,:3] * 100
    demo_starts = np.load(os.path.join(pt_data_path, "demo_starts.npy"))
    demo_ends = np.load(os.path.join(pt_data_path, "demo_ends.npy"))
    # traj_path = os.path.join("/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/20_crop_pour_can_new/Pour can into a cup9/data")
    # files = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
    env.reset()
    for i in range(len(demo_starts)):
        env.reset()
        start = demo_starts[i]
        end = demo_ends[i]
        traj_action = actions[start:end]
        demo_obs =obses[start:end]
        for step in range(traj_action.shape[0]):
            step_action = np.array(traj_action[step])
            # step_action[:3] = step_action[:3] * 100
            obs, reward, done, info = env.step(step_action)
            # cv2.imshow("test", np.transpose(obs, (1, 2, 0))[:,:,::-1])
            # cv2.imshow("test", np.transpose(np.array(demo_obs[step]), (1, 2, 0))[:,:,::-1])
            # cv2.waitKey(1)
            print("reward", reward)
            print("done", done)
            # print("info", info)
    env.close()