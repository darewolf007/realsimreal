import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import os
import agent_policy.few_shot_RL.policy_utils as policy_utils
from agent_policy.few_shot_RL.encoder import make_encoder
from agent_policy.few_shot_RL.data_augs import random_crop, center_crop, no_aug, batch_center_crop

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
        clip=None,
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
            clip=clip,
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

    def forward(self, obs, task_text=None, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        if isinstance(obs, list):
            pixel_code = self.encoder(obs[0], detach=detach_encoder)
            obs = torch.cat([pixel_code, obs[1]], dim=1)
        else:
            obs = self.encoder(obs, task_text=task_text, detach=detach_encoder)

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
        clip=None
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
            clip=clip,
        )

        trunk_input_dim = self.encoder.feature_dim

        self.Q1 = QFunction(trunk_input_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(trunk_input_dim, action_shape[0], hidden_dim)

        self.outputs = dict()

    def forward(self, obs, action, detach_encoder=False, task_text=None):
        # detach_encoder allows to stop gradient propagation to encoder
        if isinstance(obs, list):
            pixel_code = self.encoder(obs[0], detach=detach_encoder)
            obs = torch.cat([pixel_code, obs[1]], dim=1)
        else:
            obs = self.encoder(obs, task_text=task_text, detach=detach_encoder)

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
                
        self.clip, preprocess = clip.load("ViT-B/32", device=device)

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
            clip=self.clip,
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
            clip=self.clip,
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
            clip=self.clip,
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

    def select_action(self, obs, task_text=None):
        with torch.no_grad():
            obs = self.obs_to_torch(obs)
            mu, _, _, _ = self.actor(obs, task_text=task_text, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs, task_text=None):
        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)

        with torch.no_grad():
            obs = self.obs_to_torch(obs)
            mu, pi, _, _ = self.actor(obs, task_text=task_text, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step, task_text=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, task_text=task_text)

            # Action perturbation
            if self.action_noise is not None:
                noise = torch.randn_like(policy_action) * self.action_noise
                policy_action = torch.clip(policy_action + noise, -1, 1)

            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action, task_text=task_text)

            if self.v_clip_low is not None:
                target_Q1 = target_Q1.clamp(self.v_clip_low, self.v_clip_high)
                target_Q2 = target_Q2.clamp(self.v_clip_low, self.v_clip_high)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi

            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder, task_text=task_text
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

    def update_actor_and_alpha(self, obs, L, step, task_text=None):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, task_text=task_text, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, task_text=task_text, detach_encoder=True)

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
            policy_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def update_critic_only(self, replay_buffer, L, step, ema=False, translate=False):
        complex_t = "complex" in self.pretrain_mode
        obs, action, reward, next_obs, not_done, vic_pairs = replay_buffer.sample_vic(
            translate=translate, complex_augmentations=complex_t
        )
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.critic_target_update_freq == 0:
            policy_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            policy_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            policy_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def update_sac(self, L, step, obs, action, reward, next_obs, not_done, task_text=None):
        if step % self.log_interval == 0:
            L.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step, task_text=task_text)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, task_text=task_text)

        if step % self.critic_target_update_freq == 0:
            policy_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            policy_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            policy_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def update(self, replay_buffer, L, step, demo_density=None, task_text=None):
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

class E2CSacAgent(RadSacAgent):
    def update_e2c(self, replay_buffer, L, step, num_updates, init=False, mse_tol=None):
        for i in range(num_updates):
            (
                obs,
                action,
                next_obs,
                obs_non_crop,
                next_obs_non_crop,
            ) = replay_buffer.sample_e2c()
            dkl, mse, ref_kl, predict = self.e2c(
                obs, action, next_obs, obs_non_crop, next_obs_non_crop
            )
            loss = dkl + mse * 128 * 128 * 6 + ref_kl

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
                    L._sw.add_image(
                        folder + "GT_1",
                        next_obs_non_crop[0][:3].detach().cpu().numpy(),
                        global_step=i,
                    )
                    L._sw.add_image(
                        folder + "Predicted_1",
                        predict[0][:3].detach().cpu().numpy().clip(0, 1),
                        global_step=i,
                    )
                    L._sw.add_image(
                        folder + "GT_2",
                        next_obs_non_crop[0][3:].detach().cpu().numpy(),
                        global_step=i,
                    )
                    L._sw.add_image(
                        folder + "Predicted_2",
                        predict[0][3:].detach().cpu().numpy().clip(0, 1),
                        global_step=i,
                    )

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

            if step % 100 == 0:
                L._sw.add_image(
                    folder + "GT_1",
                    next_obs_non_crop[0][:3].detach().cpu().numpy(),
                    global_step=step,
                )
                L._sw.add_image(
                    folder + "Predicted_1",
                    predict[0][:3].detach().cpu().numpy().clip(0, 1),
                    global_step=step,
                )
                L._sw.add_image(
                    folder + "GT_2",
                    next_obs_non_crop[0][3:].detach().cpu().numpy(),
                    global_step=step,
                )
                L._sw.add_image(
                    folder + "Predicted_2",
                    predict[0][3:].detach().cpu().numpy().clip(0, 1),
                    global_step=step,
                )

    def update(self, replay_buffer, L, step, demo_density=None):
        if self.e2c is None:
            from e2c import E2C

            self.e2c = E2C(
                obs_shape=(6, 128, 128),
                action_dim=self.action_shape[0],
                z_dimension=16,
                crop_shape=self.obs_shape,
            ).to(self.device)
            self.e2c_optimizer = torch.optim.Adam(self.e2c.parameters(), lr=1e-4)

        if step % 300 == 0 and self.p_reward != 0:
            self.update_e2c(replay_buffer, L, step, 5000, mse_tol=1e-2)

            one_step_dist_list = []

            for i in range(len(replay_buffer.demo_starts)):
                i_start = replay_buffer.demo_starts[i]
                i_end = replay_buffer.demo_ends[i]
                demo_next_obs = replay_buffer.next_obses[i_start:i_end, :, 8:120, 8:120]
                demo_next_obs = (
                    torch.as_tensor(demo_next_obs, device=replay_buffer.device).float()
                    / 255
                )
                z_demo = (
                    self.e2c.enc(demo_next_obs)[0].unsqueeze(0).detach().cpu().numpy()
                )
                self.z_demo_cache[i] = z_demo
                one_step_dist_list.append(
                    ((z_demo[0, 1:] - z_demo[0, :-1]) ** 2).sum(axis=1).mean()
                )

            self.ref_one_step_dist = np.mean(one_step_dist_list)

        obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
            self.augs_funcs, demo_density=demo_density
        )

        if self.p_reward != 0:
            z_pred = self.e2c.enc(next_obs)[0].unsqueeze(1).detach().cpu().numpy()

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
        # No contrastive updates


class DINOE2CSacAgent(RadSacAgent):
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
            loss = dkl + mse * 384 + ref_kl

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
            # 双视角
            # image1, image2 = torch.split(obs, [3, 3], dim=1)
            # dino_emb1 = self.dino(image1)
            # dino_emb2 = self.dino(image2)
            # torch.cat([dino_emb1, dino_emb2], dim=1)
            # # 单视角
            dino_emb = self.dino(obs)
        return dino_emb
    def update(self, replay_buffer, L, step, demo_density=None, task_text=None):
        if self.e2c is None:
            from agent_policy.few_shot_RL.e2c import MLPE2C
            base_path = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(base_path, "../../pre_train/facebookresearch_dinov2_main")
            model_pth_path = os.path.join(base_path, "../../pre_train/checkpoints/dinov2_vits14_reg4_pretrain.pth")
            # self.dino = torch.hub.load(model_path, "dinov2_vits14_reg", source='local', trust_repo=True, pretrained=False).to(self.device)
            # self.dino.load_state_dict(torch.load(model_pth_path))
            # self.dino.eval()
            self.e2c = MLPE2C(
                obs_shape=(384,),
                action_dim=self.action_shape[0],
                z_dimension=16,
                crop_shape=None,
            ).to(self.device)
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

        self.update_sac(L, step, obs, action, reward, next_obs, not_done, task_text=task_text)

class DINOOnlySacAgent(RadSacAgent):
    def dino_embed(self, obs):
        with torch.no_grad():
            image1, image2 = torch.split(obs, [3, 3], dim=1)
            dino_emb1 = self.dino(image1)
            dino_emb2 = self.dino(image2)
        return torch.cat([dino_emb1, dino_emb2], dim=1)

    def update(self, replay_buffer, L, step, demo_density=None):
        if self.dino is None:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14_reg"
            ).to(self.device)

        if step == 0 and self.p_reward != 0:
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
                z_demo = dino_demo_next_obs.unsqueeze(0).detach().cpu().numpy()
                self.z_demo_cache[i] = z_demo
                one_step_dist_list.append(
                    ((z_demo[0, 1:] - z_demo[0, :-1]) ** 2).sum(axis=1).mean()
                )

            self.ref_one_step_dist = np.mean(one_step_dist_list)

        obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
            self.augs_funcs, demo_density=demo_density
        )

        if self.p_reward != 0:
            dino_next_obs = self.dino_embed(next_obs)
            z_pred = dino_next_obs.unsqueeze(1).detach().cpu().numpy()

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


class E2CILQRAgent(object):
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
        }

        for aug_name in self.data_augs.split("-"):
            if aug_name:
                assert aug_name in aug_to_func, "invalid data aug string"
                self.augs_funcs[aug_name] = aug_to_func[aug_name]

        from e2c import E2C

        self.e2c = E2C(
            obs_shape=(6, 128, 128),
            action_dim=self.action_shape[0],
            z_dimension=16,
            crop_shape=self.obs_shape,
        ).to(self.device)
        self.e2c_optimizer = torch.optim.Adam(self.e2c.parameters(), lr=1e-4)
        self.replay_buffer: policy_utils.ReplayBuffer = None
        self.training = None

    def train(self, training=True):
        self.training = training

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def obs_to_torch(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        return obs

    def save(self, model_dir, step):
        return

    def load(self, model_dir, step):
        return

    def update_e2c(self, replay_buffer, L, step, num_updates, init=False, mse_tol=None):
        for i in range(num_updates):
            (
                obs,
                action,
                next_obs,
                obs_non_crop,
                next_obs_non_crop,
            ) = replay_buffer.sample_e2c()
            dkl, mse, ref_kl, predict = self.e2c(
                obs, action, next_obs, obs_non_crop, next_obs_non_crop
            )
            loss = dkl + mse * 128 * 128 * 6 + ref_kl

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
                    L._sw.add_image(
                        folder + "GT_1",
                        next_obs_non_crop[0][:3].detach().cpu().numpy(),
                        global_step=i,
                    )
                    L._sw.add_image(
                        folder + "Predicted_1",
                        predict[0][:3].detach().cpu().numpy().clip(0, 1),
                        global_step=i,
                    )
                    L._sw.add_image(
                        folder + "GT_2",
                        next_obs_non_crop[0][3:].detach().cpu().numpy(),
                        global_step=i,
                    )
                    L._sw.add_image(
                        folder + "Predicted_2",
                        predict[0][3:].detach().cpu().numpy().clip(0, 1),
                        global_step=i,
                    )

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

            if step % 100 == 0:
                L._sw.add_image(
                    folder + "GT_1",
                    next_obs_non_crop[0][:3].detach().cpu().numpy(),
                    global_step=step,
                )
                L._sw.add_image(
                    folder + "Predicted_1",
                    predict[0][:3].detach().cpu().numpy().clip(0, 1),
                    global_step=step,
                )
                L._sw.add_image(
                    folder + "GT_2",
                    next_obs_non_crop[0][3:].detach().cpu().numpy(),
                    global_step=step,
                )
                L._sw.add_image(
                    folder + "Predicted_2",
                    predict[0][3:].detach().cpu().numpy().clip(0, 1),
                    global_step=step,
                )

    def update(self, replay_buffer, L, step, demo_density=None):
        self.update_e2c(replay_buffer, L, step, 10, mse_tol=1e-2)

    def compute_goal_embeddings(self):
        goal_indices = self.replay_buffer.demo_ends - 1
        goal_obs = self.replay_buffer.next_obses[goal_indices, :, 8:120, 8:120]
        goal_obs = torch.as_tensor(goal_obs, device=self.device).float() / 255
        z_goal = self.e2c.enc(goal_obs)[0].unsqueeze(-1)
        return z_goal.mean(dim=0, keepdim=True)

    def compute_q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx):
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.transpose(1, 2) @ V_x
        Q_u = l_u + f_u.transpose(1, 2) @ V_x
        Q_xx = l_xx + f_x.transpose(1, 2) @ V_xx @ f_x

        # Eqs (11b) and (11c).
        mu = 0
        reg = mu * torch.eye(16, device=self.device).unsqueeze(0)
        Q_ux = l_ux + f_u.transpose(1, 2) @ (V_xx + reg) @ f_x
        Q_uu = l_uu + f_u.transpose(1, 2) @ (V_xx + reg) @ f_u

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def ilqr(self, x0, n_steps=10, n_iterations=10):
        """Perform iLQR optimization"""
        # x0 is a tensor of shape (batch_size, state_dim, 1)
        x_list = [x0]
        u_list = [
            torch.zeros(1, self.action_shape[0], 1, device=self.device)
            for _ in range(n_steps)
        ]
        with torch.no_grad():
            goal = self.compute_goal_embeddings()
            I_x = torch.eye(16, device=self.device).unsqueeze(0)
            I_u = torch.eye(self.action_shape[0], device=self.device).unsqueeze(0)
            for _ in range(n_iterations):
                x_list = [x0]
                f_x_list = []
                f_u_list = []
                l_list = []
                l_x_list = []
                l_u_list = []
                l_xx_list = []
                l_ux_list = []
                l_uu_list = []
                for u in u_list:
                    x = x_list[-1]
                    A, B, offset = self.e2c.fm.get_params(x.squeeze(-1))
                    f_x_list.append(A)
                    f_u_list.append(B)
                    x_next = A @ x + B @ u + offset.unsqueeze(-1)
                    x_list.append(x_next)
                    # l = (x - goal).T @ I @ (x - goal) + u.T @ I @ u
                    l_list.append(
                        (x - goal).transpose(1, 2) @ I_x @ (x - goal)
                        + u.transpose(1, 2) @ I_u @ u
                    )
                    l_x_list.append(I_x @ (x - goal))
                    l_u_list.append(I_u @ u)
                    l_xx_list.append(I_x)
                    l_ux_list.append(
                        torch.zeros(1, self.action_shape[0], 16, device=self.device)
                    )
                    l_uu_list.append(I_u)
                x = x_list[-1]
                l_list.append((x - goal).transpose(1, 2) @ I_x @ (x - goal))
                l_x_list.append(I_x @ (x - goal))
                l_xx_list.append(I_x)

                # Backward pass
                V_x = l_x_list[-1]
                V_xx = l_xx_list[-1]

                k_list = []
                K_list = []

                for t in reversed(range(n_steps)):
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self.compute_q(
                        f_x_list[t],
                        f_u_list[t],
                        l_x_list[t],
                        l_u_list[t],
                        l_xx_list[t],
                        l_ux_list[t],
                        l_uu_list[t],
                        V_x,
                        V_xx,
                    )

                    # Eq (6).
                    k = -torch.linalg.solve(Q_uu, Q_u)
                    K = -torch.linalg.solve(Q_uu, Q_ux)
                    k_list.insert(0, k)
                    K_list.insert(0, K)

                    # Eq (11b).
                    V_x = Q_x + K.transpose(1, 2) @ Q_uu @ k
                    V_x += K.transpose(1, 2) @ Q_u + Q_ux.transpose(1, 2) @ k

                    # Eq (11c).
                    V_xx = Q_xx + K.transpose(1, 2) @ Q_uu @ K
                    V_xx += K.transpose(1, 2) @ Q_ux + Q_ux.transpose(1, 2) @ K
                    V_xx = 0.5 * (V_xx + V_xx.transpose(1, 2))  # To maintain symmetry.

                new_x_list = [x0]
                new_u_list = []

                for t in range(n_steps):
                    # Eq (12).
                    new_u_list.append(
                        torch.clip(
                            u_list[t]
                            + k_list[t]
                            + K_list[t] @ (new_x_list[t] - x_list[t]),
                            -1,
                            1,
                        )
                    )

                    # Eq (8c).
                    A, B, offset = self.e2c.fm.get_params(new_x_list[t].squeeze(-1))
                    new_x_list.append(
                        A @ new_x_list[t] + B @ new_u_list[t] + offset.unsqueeze(-1)
                    )

                u_list = new_u_list

        return u_list[0]

    def select_action(self, obs):
        with torch.no_grad():
            obs = self.obs_to_torch(obs) / 255
            x = self.e2c.enc(obs)[0]
            try:
                a = self.ilqr(x.unsqueeze(-1)).squeeze(-1)
            except:
                a = torch.rand(1, self.action_shape[0], 1, device=self.device) * 2 - 1
            return np.nan_to_num(a.cpu().numpy()).flatten().clip(-1, 1)

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)
        return self.select_action(obs)
