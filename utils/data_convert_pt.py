import os
import pickle
import torch

def convert_pickles_to_pt(data_dir, output_path):
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
        traj_path = os.path.join(data_dir, folder + "/data")
        files = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
        for file in files:
            if file.endswith(".pkl"):
                all_traj_num += 1
                file_path = os.path.join(traj_path, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    all_obses.append(data['obses'])
                    all_next_obses.append(data['next_obses'])
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
    torch.save((all_obses, all_next_obses, all_actions, all_rewards, all_not_dones), output_path)

if __name__ == "__main__":
    data_dir = "/home/haowen/hw_mine/Real_Sim_Real/data/sim_data"
    output_path = "/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pour_data.pt" 
    convert_pickles_to_pt(data_dir, output_path)