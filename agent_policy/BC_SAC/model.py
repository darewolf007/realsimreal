import torch
import copy
import numpy as np
from torch import nn
import torch.nn.functional as F

LOG_FREQ = 10000

class DINOEncoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        feature_dim,
        num_layers=2,
        num_filters=32,
        output_logits=False,
        conv_layer_norm=False,
    ):
        super().__init__()

        print("DINO ENCODER INIT")

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.fc = nn.Linear(384, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.dino = None
        self.outputs = dict()

        self.output_logits = output_logits

    def dino_embed(self, obs):
        if self.dino is None:
            global DINO
            if DINO is not None:
                self.dino = DINO
            else:
                self.dino = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14_reg"
                ).to(obs.device)
                DINO = self.dino

        with torch.no_grad():
            dino_emb1 = self.dino(obs)
        return dino_emb1

    def forward(self, obs, detach=False):
        h = self.dino_embed(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs["tanh"] = out

        return out

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_encoder/%s_hist" % k, v, step)

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
        self.encoder = DINOEncoder(
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

        self.encoder = DINOEncoder(
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

