import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import agent_policy.mine.utils as utils
from .encoder import make_encoder
from .data_augs import random_crop, center_crop, no_aug, batch_center_crop

LOG_FREQ = 10000


def gaussian_log_prob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class Actor(nn.Module):
    """MLP for actor network."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        log_std_min,
        log_std_max,
        num_layers,
        num_filters,
        conv_layer_norm=False,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
            output_logits=True,
            conv_layer_norm=conv_layer_norm,
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        trunk_input_dim = self.encoder.feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )

        self.outputs = dict()

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        # only choose scene view to predict the action
        if obs.shape[1] != self.obs_shape[0]:
            obs = obs[:,:self.obs_shape[0],:,:]
        if isinstance(obs, list):
            pixel_code = self.encoder(obs[0], detach=detach_encoder)
            obs = torch.cat([pixel_code, obs[1]], dim=1)
        else:
            obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        self.outputs["mu"] = mu
        self.outputs["std"] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            noise = None

        if compute_log_pi:
            log_pi = gaussian_log_prob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step == 0:
            return

        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_actor/%s_hist" % k, v, step)


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        assert obs.shape[0] == action.shape[0]

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employs two q-functions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        num_layers,
        num_filters,
        conv_layer_norm=False,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
            output_logits=True,
            conv_layer_norm=conv_layer_norm,
        )

        trunk_input_dim = self.encoder.feature_dim

        self.Q1 = QFunction(trunk_input_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(trunk_input_dim, action_shape[0], hidden_dim)

        self.outputs = dict()

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propagation to encoder
        if isinstance(obs, list):
            pixel_code = self.encoder(obs[0], detach=detach_encoder)
            obs = torch.cat([pixel_code, obs[1]], dim=1)
        else:
            obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step == 0:
            return

        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram("train_critic/%s_hist" % k, v, step)


class RadSacAgent(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type="pixel",
        encoder_feature_dim=32,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
        data_augs="",
        v_clip_low=None,
        v_clip_high=None,
        action_noise=None,
        pretrain_mode=None,
        conv_layer_norm=False,
        p_reward=1,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.latent_dim = latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.data_augs = data_augs

        self.v_clip_low = v_clip_low
        self.v_clip_high = v_clip_high
        self.action_noise = action_noise
        self.pretrain_mode = pretrain_mode

        self.e2c = None
        self.dino = None
        self.moco = None
        self.e2c_optimizer = None
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.encoder_feature_dim = encoder_feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.p_reward = p_reward
        self.z_demo_cache = {}
        self.ref_one_step_dist = None

        self.augs_funcs = {}

        aug_to_func = {
            "crop": random_crop,
            "no_aug": no_aug,
            "center_crop": batch_center_crop,
        }

        for aug_name in self.data_augs.split("-"):
            if aug_name:
                assert aug_name in aug_to_func, "invalid data aug string"
                self.augs_funcs[aug_name] = aug_to_func[aug_name]

        self.actor = Actor(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            actor_log_std_min,
            actor_log_std_max,
            num_layers,
            num_filters,
            conv_layer_norm=conv_layer_norm,
        ).to(device)

        self.critic = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
            conv_layer_norm=conv_layer_norm,
        ).to(device)

        self.critic_target = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
            conv_layer_norm=conv_layer_norm,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.training = None
        self.train()
        self.critic_target.train()

        self.bn = torch.nn.BatchNorm1d(encoder_feature_dim).to(device)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def obs_to_torch(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        return obs

    def select_action(self, obs):
        with torch.no_grad():
            obs = self.obs_to_torch(obs)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)

        with torch.no_grad():
            obs = self.obs_to_torch(obs)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)

            # Action perturbation
            if self.action_noise is not None:
                noise = torch.randn_like(policy_action) * self.action_noise
                policy_action = torch.clip(policy_action + noise, -1, 1)

            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)

            if self.v_clip_low is not None:
                target_Q1 = target_Q1.clamp(self.v_clip_low, self.v_clip_high)
                target_Q2 = target_Q2.clamp(self.v_clip_low, self.v_clip_high)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi

            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder
        )
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if step % self.log_interval == 0:
            L.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log("train_actor/loss", actor_loss, step)
            L.log("train_actor/target_entropy", self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )
        if step % self.log_interval == 0:
            L.log("train_actor/entropy", entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log("train_alpha/loss", alpha_loss, step)
            L.log("train_alpha/value", self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_contrastive(
        self, obs_a, obs_b, L, step, ema=False, obs=None, action=None, next_obs=None
    ):
        z_a = self.critic.encoder(obs_a)
        z_b = self.critic.encoder(obs_b)

        if self.pretrain_mode == "CURL":
            logits = self.CURL.compute_logits(z_a, z_b)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            loss = self.cross_entropy_loss(logits, labels)

            self.encoder_optimizer.zero_grad()
            self.cpc_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.cpc_optimizer.step()
        else:
            loss = None
            raise RuntimeError("Unknown pre-train mode.")

        if step % self.log_interval == 0:
            L.log("train/contrastive_loss", loss, step)
        if ema:
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def update_critic_only(self, replay_buffer, L, step, ema=False, translate=False):
        complex_t = "complex" in self.pretrain_mode
        obs, action, reward, next_obs, not_done, vic_pairs = replay_buffer.sample_vic(
            translate=translate, complex_augmentations=complex_t
        )
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def update_sac(self, L, step, obs, action, reward, next_obs, not_done):
        # only choose scene view to predict the action
        if obs.shape[1] != self.obs_shape[0]:
            obs = obs[:,:self.obs_shape[0],:,:]
            next_obs = next_obs[:,:self.obs_shape[0],:,:]
        if step % self.log_interval == 0:
            L.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def update(self, replay_buffer, L, step, demo_density=None):
        if self.encoder_type == "pixel" or self.encoder_type == "dino":
            obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
                self.augs_funcs, demo_density=demo_density
            )
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        self.update_sac(L, step, obs, action, reward, next_obs, not_done)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), "%s/actor_%s.pt" % (model_dir, step))
        torch.save(self.critic.state_dict(), "%s/critic_%s.pt" % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load("%s/actor_%s.pt" % (model_dir, step)))
        self.critic.load_state_dict(torch.load("%s/critic_%s.pt" % (model_dir, step)))


class MineSacAgent(RadSacAgent):
    def update_e2c(self, replay_buffer, L, step, num_updates, init=False, mse_tol=None):
        for i in range(num_updates):
            (
                obs,
                action,
                next_obs,
                obs_non_crop,
                next_obs_non_crop,
            ) = replay_buffer.sample_e2c()
            dino_obs = self.dino_embed(obs)
            dino_next_obs = self.dino_embed(next_obs)
            dkl, mse, ref_kl, predict = self.e2c(
                dino_obs, action, dino_next_obs, None, None
            )
            if replay_buffer.obses.shape[1] == 3:
                loss = dkl + mse * 384 + ref_kl
            elif replay_buffer.obses.shape[1] == 6:
                loss = dkl + mse * 768 + ref_kl
            else:
                raise RuntimeError("Unknown obs shape")
            self.e2c_optimizer.zero_grad()
            loss.backward()
            self.e2c_optimizer.step()

            if init:
                folder = "train_e2c_init/"
                if i % 10 == 0:
                    L._sw.add_scalar(folder + "dkl", dkl, i)
                    L._sw.add_scalar(folder + "mse", mse, i)
                    L._sw.add_scalar(folder + "ref_kl", ref_kl, i)
                    L._sw.add_scalar(folder + "loss", loss, i)

                if i % 100 == 0:
                    print(f"E2C loss: {loss}")

            if mse_tol is not None and mse.detach().cpu().item() < mse_tol:
                break

        if not init:
            folder = "train_e2c_training/"
            if step % 10 == 0:
                L._sw.add_scalar(folder + "updates", i + 1, step)
                L._sw.add_scalar(folder + "dkl", dkl, step)
                L._sw.add_scalar(folder + "mse", mse, step)
                L._sw.add_scalar(folder + "ref_kl", ref_kl, step)
                L._sw.add_scalar(folder + "loss", loss, step)

    def dino_embed(self, obs):
        with torch.no_grad():
            if obs.shape[1] == 3:
                return self.dino(obs)
            else:
                image1, image2 = torch.split(obs, [3, 3], dim=1)
                dino_emb1 = self.dino(image1)
                dino_emb2 = self.dino(image2)
                return torch.cat([dino_emb1, dino_emb2], dim=1)

    def update(self, replay_buffer, L, step, demo_density=None):
        if self.e2c is None:
            from .e2c import MLPE2C
            if replay_buffer.obses.shape[1] == 3:
                self.e2c = MLPE2C(
                    obs_shape=(384,),
                    action_dim=self.action_shape[0],
                    z_dimension=16,
                    crop_shape=None,
                ).to(self.device)
            elif replay_buffer.obses.shape[1] == 6:
                self.e2c = MLPE2C(
                    obs_shape=(768,),
                    action_dim=self.action_shape[0],
                    z_dimension=16,
                    crop_shape=None,
                ).to(self.device)
            else:
                raise RuntimeError("Unknown obs shape")
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14_reg"
            ).to(self.device)
            self.e2c_optimizer = torch.optim.Adam(self.e2c.parameters(), lr=1e-4)

        if step % 300 == 0 and self.p_reward != 0:
            self.update_e2c(replay_buffer, L, step, 1000, mse_tol=0.2)

            one_step_dist_list = []

            for i in range(len(replay_buffer.demo_starts)):
                i_start = replay_buffer.demo_starts[i]
                i_end = replay_buffer.demo_ends[i]
                demo_next_obs = replay_buffer.next_obses[i_start:i_end, :, 8:120, 8:120]
                demo_next_obs = (
                    torch.as_tensor(demo_next_obs, device=replay_buffer.device).float()
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

        if self.encoder_type == "pixel":
            obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
                self.augs_funcs, demo_density=demo_density
            )
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
                self.augs_funcs, demo_density=demo_density
            )

        if self.p_reward != 0:
            dino_next_obs = self.dino_embed(next_obs)
            z_pred = self.e2c.enc(dino_next_obs)[0].unsqueeze(1).detach().cpu().numpy()

            min_dist = np.ones(len(next_obs)) * 10000
            discount_power = np.zeros(len(next_obs))
            for i in range(len(replay_buffer.demo_starts)):
                i_start = replay_buffer.demo_starts[i]
                i_end = replay_buffer.demo_ends[i]
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
            if step % self.log_interval == 0:
                L.log(
                    "train/avg_discount",
                    (discount_power * reward_mask).sum()
                    / reward_mask.astype(int).sum(),
                    step,
                )
                L.log(
                    "train/num_additional_reward",
                    (min_dist < self.ref_one_step_dist).sum(),
                    step,
                )

            reward += torch.as_tensor(
                additional_reward, device=reward.device
            ).unsqueeze(1)

        self.update_sac(L, step, obs, action, reward, next_obs, not_done)

    def save(self, model_dir, step):
        super().save(model_dir, step)
        if self.e2c is not None:
            torch.save(
                self.e2c.state_dict(), "%s/e2c_%s.pt" % (model_dir, step)
            )
            torch.save(
                self.dino.state_dict(), "%s/dino_%s.pt" % (model_dir, step)
            )

    def load(self, model_dir, step):
        super().load(model_dir, step)
        self.e2c.load_state_dict(
            torch.load("%s/e2c_%s.pt" % (model_dir, step), map_location=self.device)
        )
        self.dino.load_state_dict(
            torch.load("%s/dino_%s.pt" % (model_dir, step), map_location=self.device)
        )