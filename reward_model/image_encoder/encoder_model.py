import os
import torch
from torch import nn
from min_vit import MinVit
from dataclasses import dataclass, field

@dataclass
class VitEncoderConfig:
    patch_size: int = 8
    depth: int = 3
    embed_dim: int = 128
    num_heads: int = 4
    act_layer = nn.GELU
    stride: int = -1
    embed_style: str = "embed1"
    embed_norm: int = 0

class VitEncoder(nn.Module):
    def __init__(self, obs_shape: list[int], cfg: VitEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
        )

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim
        self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def forward(self, obs, flatten=True) -> torch.Tensor:
        assert obs.max() > 5
        obs = obs / 255.0 - 0.5
        feats: torch.Tensor = self.vit.forward(obs)
        if flatten:
            feats = feats.flatten(1, 2)
        return feats

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

        self.fc = nn.Linear(768, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.dino = None
        self.outputs = dict()

        self.output_logits = output_logits

    def dino_embed(self, obs):
        if self.dino is None:
            base_path = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(base_path, "../../pre_train/dinov2")
            model_pth_path = os.path.join(base_path, "../../pre_train/dinov2_vits14.pth")
            self.dino = torch.hub.load(model_path, 'dinov2_vits14', source='local', pretrained=False).to(obs.device)
            self.dino.load_state_dict(torch.load(model_pth_path))
            # self.dino = torch.hub.load(
            #     "facebookresearch/dinov2", "dinov2_vits14_reg"
            # ).to(obs.device)
        with torch.no_grad():
            image1, image2 = torch.split(obs, [3, 3], dim=1)
            dino_emb1 = self.dino(image1)
            dino_emb2 = self.dino(image2)
        return torch.cat([dino_emb1, dino_emb2], dim=1)

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


if __name__ == "__main__":
    obs_shape = [3, 64, 64]
    encoder = DINOEncoder(obs_shape, 256)
    print(encoder)
    test_image = torch.randn(2, 6, 448, 448)
    feat = encoder.dino_embed(test_image)
