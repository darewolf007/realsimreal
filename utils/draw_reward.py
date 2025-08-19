import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

reward_sparse = np.loadtxt('/home/haowen/hw_mine/Real_Sim_Real/sparse_reward_collect_list_0.csv', delimiter=',')
reward_dense = np.loadtxt('/home/haowen/hw_mine/Real_Sim_Real/reward_collect_list_0.csv', delimiter=',')
reward_ours = np.loadtxt('/home/haowen/hw_mine/Real_Sim_Real/lane_reward_collect_list_0.csv', delimiter=',')

# 平滑处理（可选）
# reward_sparse = gaussian_filter1d(reward_sparse, sigma=1)
# reward_dense = gaussian_filter1d(reward_dense, sigma=1)
# reward_ours = gaussian_filter1d(reward_ours, sigma=1)

# 横轴时间步
timesteps = np.arange(len(reward_dense))

# 绘图
fig, axs = plt.subplots(3, 1, figsize=(7, 5), sharex=True)

axs[1].plot(timesteps, reward_dense, color='black', linewidth=0.8)
axs[1].set_ylabel('Reward')
axs[1].set_title('Dense (Manual Shaping)')
axs[1].spines[['top', 'right']].set_visible(False)

axs[0].plot(timesteps, reward_sparse, color='black', linewidth=0.8)
axs[0].set_ylabel('Reward')
axs[0].set_title('Sparse (Manual Only)')
axs[0].spines[['top', 'right']].set_visible(False)



axs[2].plot(timesteps, reward_ours, color='black', linewidth=0.8)
axs[2].set_ylabel('Reward')
axs[2].set_xlabel('Timestep')
axs[2].set_title('Ours')
axs[2].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("reward_trends_split_axes.png", dpi=300)
plt.show()

