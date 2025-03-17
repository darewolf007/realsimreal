import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
from torch.nn import functional as F
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from agent_policy.diffusion.policy_util.diffusion_policy import ConditionalUnet1D

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        # 从参数中获取关键维度和超参数
        self.observation_horizon = args_override['observation_horizon']  # 观察历史长度
        self.action_horizon = args_override['action_horizon']           # 动作窗口大小
        self.prediction_horizon = args_override['prediction_horizon']   # 预测长度
        self.num_inference_timesteps = args_override['num_inference_timesteps']  # 推理步数
        self.ema_power = args_override['ema_power']                    # EMA 指数
        self.lr = args_override['lr']                                  # 学习率
        self.weight_decay = 0                                          # 权重衰减

        # 定义观察和动作维度
        self.obs_dim = args_override['obs_dim']  # 环境状态向量的维度 n
        self.ac_dim = args_override['action_dim']  # 动作维度

        # 初始化去噪网络
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon  # 全局条件维度
        )

        # 网络模块
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'noise_pred_net': noise_pred_net
            })
        })
        nets = nets.float().cuda()

        # 设置 EMA（指数移动平均）
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # 初始化噪声调度器
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        # 打印参数数量
        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def __call__(self, obs, actions=None, is_pad=None):
        B = obs.shape[0]  # 批量大小
        if actions is not None:  # 训练时
            nets = self.nets
            # obs 形状为 (B, observation_horizon, obs_dim)，展平为全局条件
            obs_cond = obs.reshape(B, -1)  # (B, observation_horizon * obs_dim)

            # 生成噪声
            noise = torch.randn(actions.shape, device=obs_cond.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()

            # 向动作添加噪声
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # 预测噪声
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # 计算损失
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {'l2_loss': loss, 'loss': loss}
            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else:  # 推理时
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            nets = self.nets if self.ema is None else self.ema.averaged_model

            # 处理观察条件
            obs_cond = obs.reshape(B, -1)  # (B, observation_horizon * obs_dim)

            # 从高斯噪声初始化动作
            noisy_action = torch.randn((B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            # 设置推理步数
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            # 去噪过程
            for k in self.noise_scheduler.timesteps:
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status