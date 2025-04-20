import numpy as np

file_path = '/home/haowen/hw_mine/Real_Sim_Real/data/real_data/easy_task/pick_place_apple_bowl/1/traj/traj_55.npy'
data = np.load(file_path, allow_pickle=True)
print("原始数据：", data)
data[-1] = 0
np.save(file_path, data)
data_check = np.load(file_path, allow_pickle=True)
print("修改验证：", data_check)
