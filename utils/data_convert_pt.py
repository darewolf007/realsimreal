import os
import pickle
import torch
import numpy as np
import sys
sys.path.insert(0, os.getcwd())
from utils.image_util import resize_image
import shutil
import matplotlib.pyplot as plt

def convert_pickles_to_pt(data_dir, output_path, crop):
    all_obses = []
    all_next_obses = []
    all_actions = []
    all_rewards = []
    all_not_dones = []
    all_subtask_id = []
    all_now_qpos = []
    traj_start = [0]
    all_traj_num = 0
    subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for folder in subfolders:
        print(folder)
        traj_path = os.path.join(data_dir, folder + "/data")
        files = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
        for file in files:
            if file.endswith(".pkl"):
                all_traj_num += 1
                file_path = os.path.join(traj_path, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    all_obses.append(np.transpose(resize_image(data['obses'], crop), (2, 0, 1)))
                    all_next_obses.append(np.transpose(resize_image(data['next_obses'], crop), (2, 0, 1)))
                    # data['actions'][3:-1] = np.radians(data['actions'][3:-1])
                    data['actions'][3:-1] = np.array([0,0,0])
                    all_actions.append(data['actions'])
                    if data['rewards'] == 0:
                        all_rewards.append(-1)
                    else:
                        if not data['not_dones']:
                            all_rewards.append(data['rewards'])
                        else:
                            # all_rewards.append(-1)
                            all_rewards.append(data['rewards'])
                    all_not_dones.append(data['not_dones'])
                    all_subtask_id.append(data['subtask_id'])
                    all_now_qpos.append(data['now_qpos'])
        traj_start.append(all_traj_num)
    start = 0
    end = all_traj_num
    demo_starts = traj_start[:-1]
    demo_ends = traj_start[1:]
    
    all_obses = torch.tensor(all_obses)
    all_next_obses = torch.tensor(all_next_obses)
    all_actions = torch.tensor(all_actions)
    all_rewards = torch.tensor(all_rewards)
    all_not_dones = torch.tensor(all_not_dones)
    print("angle", all_actions[:, 3:-1].max())
    print("action", all_actions[:, :3].max())
    pt_file_name = str(start) + "_" + str(end) +".pt"
    torch.save((all_obses, all_next_obses, all_actions, all_rewards, all_not_dones), output_path + pt_file_name)
    np.save(os.path.join(output_path, "demo_ends.npy"), demo_ends)
    np.save(os.path.join(output_path, "demo_starts.npy"), demo_starts)

def convert_real_to_pt(data_dir, output_path):
    all_obses = []
    all_next_obses = []
    all_actions = []
    all_rewards = []
    all_not_dones = []
    all_subtask_id = []
    all_now_qpos = []
    traj_start = [0]
    all_traj_num = 0
    subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for folder in subfolders:
        print(folder)
        traj_path = os.path.join(data_dir, folder + "/data")
        files = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
        for file in files:
            if file.endswith(".pkl"):
                all_traj_num += 1
                file_path = os.path.join(traj_path, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    all_obses.append(np.transpose(resize_image(data['obses'], 1/6), (2, 0, 1)))
                    all_next_obses.append(np.transpose(resize_image(data['next_obses'], 1/6), (2, 0, 1)))
                    all_actions.append(data['actions'])
                    all_rewards.append(data['rewards'])
                    all_not_dones.append(data['not_dones'])
                    all_subtask_id.append(data['subtask_id'])
                    all_now_qpos.append(data['now_qpos'])
        traj_start.append(all_traj_num)
    start = 0
    end = all_traj_num
    demo_starts = traj_start[:-1]
    demo_ends = traj_start[1:]
    
    all_obses = torch.tensor(all_obses)
    all_next_obses = torch.tensor(all_next_obses)
    all_actions = torch.tensor(all_actions)
    all_rewards = torch.tensor(all_rewards)
    all_not_dones = torch.tensor(all_not_dones)
    pt_file_name = str(start) + "_" + str(end) +".pt"
    torch.save((all_obses, all_next_obses, all_actions, all_rewards, all_not_dones), output_path + pt_file_name)
    np.save(os.path.join(output_path, "demo_ends.npy"), demo_starts)
    np.save(os.path.join(output_path, "demo_starts.npy"), demo_ends)

def check_data_quality(data_dir, crop):
    all_obses = []
    all_next_obses = []
    all_actions = []
    all_rewards = []
    all_not_dones = []
    all_subtask_id = []
    all_now_qpos = []
    traj_start = [0]
    all_traj_num = 0
    subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for folder in subfolders:
        print(folder)
        traj_path = os.path.join(data_dir, folder + "/data")
        files = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
        for file in files:
            if file.endswith(".pkl"):
                all_traj_num += 1
                file_path = os.path.join(traj_path, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    all_obses.append(np.transpose(resize_image(data['obses'], crop), (2, 0, 1)))
                    all_next_obses.append(np.transpose(resize_image(data['next_obses'], crop), (2, 0, 1)))
                    # data['actions'][3:-1] = np.radians(data['actions'][3:-1])
                    data['actions'][3:-1] = np.array([0,0,0])
                    all_actions.append(data['actions'])
                    if data['rewards'] == 0:
                        all_rewards.append(-1)
                    else:
                        if not data['not_dones']:
                            all_rewards.append(data['rewards'])
                        else:
                            # all_rewards.append(-1)
                            all_rewards.append(data['rewards'])
                    all_not_dones.append(data['not_dones'])
                    all_subtask_id.append(data['subtask_id'])
                    all_now_qpos.append(data['now_qpos'])
        traj_start.append(all_traj_num)
    actions = np.array(all_actions) * 100
    num_dims = actions.shape[1]  # 获取 action 维度
    means = np.mean(actions, axis=0)
    stds = np.std(actions, axis=0)
    for i in range(len(means)):
        print(f"Action Dim {i}: Mean = {means[i]:.3f}, Std = {stds[i]:.3f}")
    # for i in range(num_dims):
    #     plt.figure()
    #     plt.hist(actions[:, i], bins=50, density=True, alpha=0.7, color='blue')
    #     plt.title(f'Action Dimension {i} Distribution')
    #     plt.xlabel('Action Value')
    #     plt.ylabel('Density')
    #     plt.grid()
    # plt.show()

if __name__ == "__main__":
    data_name = "dense_banana"
    if "crop" in data_name:
        crop = 1/6
    else:
        crop = 1/12
    data_dir = "/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/dense/" + data_name
    output_path = "/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pt_data/" 
    check_data_quality(data_dir, crop)
    # pt_output_path = output_path + data_name
    # if os.path.exists(pt_output_path):
    #     shutil.rmtree(pt_output_path)
    # os.mkdir(pt_output_path)
    # convert_pickles_to_pt(data_dir, pt_output_path + "/", crop)