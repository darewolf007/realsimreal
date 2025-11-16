import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.lane_utils import ReplayBuffer, soft_update_params
from .utils.encoder import make_encoder
from .utils.data_augs import random_crop, center_crop, no_aug, batch_center_crop
from .utils.e2c import MLPE2C

class DINOE2CSacAgent:
    def __init__(self, replay_buffer_load_dir, reward_obs_shape, action_shape, device, reward_model_dir, dino_model_dir, p_reward=1.0):
        self.replay_buffer = ReplayBuffer(
        obs_shape=reward_obs_shape,
        action_shape=action_shape,
        capacity=500,
        batch_size=1,
        device=device,
        image_size=reward_obs_shape[1],
        load_dir=replay_buffer_load_dir,
        keep_loaded=False,
        )
        self.action_shape = action_shape
        self.device = device
        self.p_reward = p_reward
        self.e2c = None
        self.dino = None
        self.e2c_optimizer = None
        self.z_demo_cache = {}
        self.ref_one_step_dist = 0.0
        self.model_dir = reward_model_dir
        self.dino_model_dir = dino_model_dir

    def dino_embed(self, obs):
        if obs.shape[2] == 128:
            obs = obs[:, :, 8:120, 8:120]
        with torch.no_grad():
            if obs.shape[1] == 3:
                return self.dino(obs)
            else:
                image1, image2 = torch.split(obs, [3, 3], dim=1)
                dino_emb1 = self.dino(image1)
                dino_emb2 = self.dino(image2)
                return torch.cat([dino_emb1, dino_emb2], dim=1)

    def get_reward(self, next_obs, not_done):
        next_obs = (torch.as_tensor(next_obs, device=self.replay_buffer.device).float()/255)
        if self.e2c is None:
            if self.replay_buffer.obses.shape[1] == 3:
                self.e2c = MLPE2C(
                    obs_shape=(384,),
                    action_dim=self.action_shape[0],
                    z_dimension=16,
                    crop_shape=None,
                ).to(self.device)
            elif self.replay_buffer.obses.shape[1] == 6:
                self.e2c = MLPE2C(
                    obs_shape=(768,),
                    action_dim=self.action_shape[0],
                    z_dimension=16,
                    crop_shape=None,
                ).to(self.device)
            else:
                raise RuntimeError("Unknown obs shape")
            self.dino = torch.hub.load(self.dino_model_dir, "dinov2_vits14_reg", source='local', trust_repo=True, pretrained=False).to(self.device)
            self.dino.load_state_dict(torch.load(self.dino_model_dir + '/dinov2_vits14_reg4_pretrain.pth'))
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False
            # self.dino = torch.hub.load(
            #     "facebookresearch/dinov2", "dinov2_vits14_reg"
            # ).to(self.device)
            self.e2c.load_state_dict(
            torch.load(self.model_dir, map_location=self.device))
            self.e2c.eval()
            for param in self.e2c.parameters():
                param.requires_grad = False
            one_step_dist_list = []
            for i in range(len(self.replay_buffer.demo_starts)):
                i_start = self.replay_buffer.demo_starts[i]
                i_end = self.replay_buffer.demo_ends[i]
                demo_next_obs = self.replay_buffer.next_obses[i_start:i_end, :, 8:120, 8:120]
                demo_next_obs = (
                    torch.as_tensor(demo_next_obs, device=self.replay_buffer.device).float()
                    / 255
                )
                dino_demo_next_obs = self.dino_embed(demo_next_obs)
                z_demo = (
                    self.e2c.enc(dino_demo_next_obs)[0]
                    .unsqueeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                self.z_demo_cache[i] = z_demo
                one_step_dist_list.append(
                    ((z_demo[0, 1:] - z_demo[0, :-1]) ** 2).sum(axis=1).mean()
                )
            self.ref_one_step_dist = np.mean(one_step_dist_list)

        dino_next_obs = self.dino_embed(next_obs)
        z_pred = self.e2c.enc(dino_next_obs)[0].unsqueeze(1).detach().cpu().numpy()

        min_dist = np.ones(len(next_obs)) * 10000
        discount_power = np.zeros(len(next_obs))
        for i in range(len(self.replay_buffer.demo_starts)):
            i_start = self.replay_buffer.demo_starts[i]
            i_end = self.replay_buffer.demo_ends[i]
            z_demo = self.z_demo_cache[i]
            z_dist = ((z_demo - z_pred) ** 2).sum(axis=2)
            z_dist_min = z_dist.min(axis=1)
            update_min = z_dist_min < min_dist
            min_dist[update_min] = z_dist_min[update_min]
            discount_power[update_min] = (
                z_dist.shape[1] - z_dist.argmin(axis=1)[update_min]
            )

        demo_reward_discount = 0.98
        reward_mask = np.logical_and(
            min_dist < self.ref_one_step_dist,
            not_done.detach().cpu().numpy().flatten(),
        )
        additional_reward = (
            np.power(demo_reward_discount, discount_power)
            * reward_mask
            * self.p_reward
        )
        return torch.as_tensor(additional_reward, device=next_obs.device).unsqueeze(1)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "/home/haowen/hw_mine/Real_Sim_Real/experiments/Pick up apple and place it to the bowl/Pick up apple and place it to the bowl-PickApplePlaceBowl-LaNE-test-2025-04-24-11-43-06/model/e2c_250000.pt"
    reward_obs_shape = (3, 128, 128)
    action_shape = (7,)
    replay_buffer_load_dir = "/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pt_data/Pick up apple and place it to the bowl"
    reward_model = DINOE2CSacAgent(replay_buffer_load_dir, reward_obs_shape, action_shape, device, model_dir, dino_model_dir=None)
    obses, actions, next_obses, obs_non_crop, next_obs_non_crop = reward_model.replay_buffer.sample_e2c()
    reward = reward_model.get_reward(np.array([0]), next_obses, torch.tensor([1]))