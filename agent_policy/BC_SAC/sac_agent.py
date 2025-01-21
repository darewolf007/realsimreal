import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Actor, Critic
from .utils import random_crop, center_crop, no_aug, batch_center_crop, soft_update_params

LOG_FREQ = 10000

class SacAgent(object):
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
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def update_critic_only(self, replay_buffer, L, step, ema=False, translate=False):
        complex_t = "complex" in self.pretrain_mode
        obs, action, reward, next_obs, not_done, vic_pairs = replay_buffer.sample_vic(
            translate=translate, complex_augmentations=complex_t
        )
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def update_sac(self, L, step, obs, action, reward, next_obs, not_done):
        if step % self.log_interval == 0:
            L.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
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