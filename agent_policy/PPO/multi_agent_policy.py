from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import wandb
import imageio
from pathlib import Path


# =============================================================================
# 1. 工具类 (VideoRecorder, DictArray, Logger)
# =============================================================================

class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = Path(root_dir) / 'eval_video'
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            # 默认录制图像数据，obs 应该是一个 HWC 的 numpy array
            self.frames.append(obs)

    def save(self, file_name):
        if self.enabled and len(self.frames) > 0:
            path = self.save_dir / file_name
            # 使用 imageio 保存视频
            imageio.mimsave(str(path), self.frames, fps=self.fps, codec="libx264")
            # 清空缓存
            self.frames = []


class DictArray(object):
    """用于在 PPO Buffer 中存储字典类型的观测数据"""

    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                             torch.uint8 if v.dtype == np.uint8 else
                             torch.int16 if v.dtype == np.int16 else
                             torch.int32 if v.dtype == np.int32 else
                             v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer:
            self.writer.close()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# 2. 配置类 (Args) - 在这里配置单/双视角
# =============================================================================

@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    use_wandb: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    wandb_group: str = "PPO"
    capture_video: bool = True
    save_model: bool = True
    save_path: str = "./runs"  # 默认保存路径
    evaluate: bool = False
    checkpoint: Optional[str] = None
    render_mode: str = "all"

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    include_state: bool = True
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 16  # 示例默认值
    num_eval_envs: int = 1
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    reward_scale: float = 1.0
    eval_freq: int = 25
    save_train_video_freq: Optional[int] = None
    finite_horizon_gae: bool = False
    image_size: int = 128

    # --- [关键配置] 视角选择 ---
    # 单视角: ["rgb"]
    # 双视角: ["rgb", "rgb_2"] 或 ["base_camera", "hand_camera"]
    train_camera_name: List[str] = field(default_factory=lambda: ["rgb", "rgb_2"])

    # Runtime variables
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# =============================================================================
# 3. 网络结构 (NatureCNN & PPOAgent) - 动态支持多输入
# =============================================================================

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()
        extractors = {}
        self.out_features = 0
        feature_size = 256

        # CNN 构建辅助函数
        def make_cnn(input_shape):
            in_channels = input_shape[-1]
            return nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
                nn.Flatten(),
            )

        # 遍历 sample_obs 中的所有 key
        print("Network Structure:")
        for key, value in sample_obs.items():
            if key == "state":
                # 状态向量处理
                state_size = value.shape[-1]
                extractors[key] = nn.Linear(state_size, 256)
                self.out_features += 256
                print(f"  - Branch [state]: Linear({state_size} -> 256)")
            else:
                # 图像处理 (假设 H,W,C 格式)
                cnn = make_cnn(value.shape)
                with torch.no_grad():
                    # 模拟一次前向传播计算 Flatten 维度
                    # [B, H, W, C] -> [B, C, H, W]
                    dummy_input = value.float().permute(0, 3, 1, 2).cpu()
                    n_flatten = cnn(dummy_input).shape[1]

                extractors[key] = nn.Sequential(
                    cnn,
                    nn.Linear(n_flatten, feature_size),
                    nn.ReLU()
                )
                self.out_features += feature_size
                print(f"  - Branch [{key}]: CNN -> Linear({n_flatten} -> {feature_size})")

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            # 这里的 observations 是从 Buffer 取出的数据
            if key in observations:
                obs = observations[key]
                if key != "state":
                    # 图像: 归一化并转置 [B, H, W, C] -> [B, C, H, W]
                    obs = obs.float().permute(0, 3, 1, 2) / 255.0
                encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class PPOAgent(nn.Module):
    def __init__(self, action_shape, obs_shape=None, state_shape=(7,), train_camera_name=["rgb"]):
        super().__init__()

        # 根据 train_camera_name 构建虚拟的 sample_obs 用于初始化网络
        sample_obs = {}
        for key in train_camera_name:
            # 假设所有图像都是 obs_shape (例如 128x128x3)
            # 如果你有不同分辨率的相机，需要这里做特殊判断
            sample_obs[key] = torch.zeros(1, *obs_shape)

        # 如果需要状态，这里需要根据 env 配置
        # sample_obs["state"] = torch.zeros(1, *state_shape)

        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(action_shape)), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(action_shape)) * -0.5)

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# =============================================================================
# 4. 主训练函数 (ppo_train_main)
# =============================================================================

def ppo_train_main(agent, cfg, env, test_env, post_process_fn=None, reward_fn=None, action_pre_process_fn=None):
    """
    Args:
        agent: PPOAgent 实例
        cfg: 包含 Args 的配置对象 (cfg.agent 应该是 Args 实例)
        env: 训练环境 (VectorEnv)
        test_env: 测试环境
        post_process_fn: 将 raw obs 转换为网络需要的 dict 格式
        action_pre_process_fn: 处理动作 (如 rescale)
    """
    args = cfg.agent

    # 动态计算运行时参数
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{cfg.task_name}__{args.seed}__{int(time.time())}"

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Logger 初始化
    logger = None
    video_recorder = VideoRecorder(Path(args.save_path) / run_name)

    if not args.evaluate:
        print(f"--- Training Start ---")
        print(f"Observation Keys: {args.train_camera_name}")

        if args.use_wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=run_name,
                save_code=False,
                group=args.wandb_group,
                tags=["ppo", "flexible_view"]
            )
        writer = SummaryWriter(f"{args.save_path}/{run_name}")
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])), )
        logger = Logger(log_wandb=args.use_wandb, tensorboard=writer)
    else:
        print("--- Evaluation Mode ---")

    # -------------------------------------------------------------------------
    # 核心修改：动态创建 Replay Buffer (DictArray)
    # -------------------------------------------------------------------------
    obs_space_dict = {}
    for key in args.train_camera_name:
        obs_space_dict[key] = spaces.Box(
            low=0, high=255,
            shape=(args.image_size, args.image_size, 3),
            dtype=np.uint8
        )
    if args.include_state:
        # 这里假设 state 维度为 7，如果不同需调整
        obs_space_dict["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    simple_obs_space = spaces.Dict(obs_space_dict)

    # 初始化 Buffer
    obs = DictArray((args.num_steps, args.num_envs), simple_obs_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + env.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # -------------------------------------------------------------------------
    # 训练准备
    # -------------------------------------------------------------------------
    global_step = 0
    start_time = time.time()

    # Reset Environment
    next_obs = env.reset()
    next_obs = post_process_fn(next_obs)  # [Check] 必须返回 dict 且包含 args.train_camera_name

    eval_obs = test_env.reset()
    eval_obs = post_process_fn(eval_obs)
    next_done = torch.zeros(args.num_envs, device=device)

    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)
    running_episode_rewards = torch.zeros(args.num_envs, device=device)
    running_episode_lengths = torch.zeros(args.num_envs, device=device)

    # -------------------------------------------------------------------------
    # 主循环
    # -------------------------------------------------------------------------
    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        # === Evaluation Block ===
        if iteration % args.eval_freq == 1:
            print("Evaluating...")
            stime = time.perf_counter()
            eval_obs = test_env.reset()
            eval_obs = post_process_fn(eval_obs)
            eval_metrics = defaultdict(list)

            # 初始化录制: 使用列表中的第一个相机视角
            primary_cam = args.train_camera_name[0]
            video_recorder.init(eval_obs[primary_cam].cpu().numpy()[0], enabled=True)

            # 简化版 Evaluation Loop
            for _ in range(1):
                for _ in range(args.num_eval_steps):
                    with torch.no_grad():
                        eval_action = action_pre_process_fn(agent.get_action(eval_obs, deterministic=True))
                        eval_obs, eval_rew, eval_done, eval_infos = test_env.step(eval_action)
                        eval_obs = post_process_fn(eval_obs)
                        video_recorder.record(eval_obs[primary_cam].cpu().numpy()[0])

                        if "final_info" in eval_infos:
                            for k, v in eval_infos["final_info"]["episode"].items():
                                eval_metrics[k].append(v)
                video_recorder.save(f'epoch_{iteration}.mp4')

            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger: logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"  eval_{k}_mean={mean:.4f}")

            if logger:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)

        # === Save Model ===
        if args.save_model and iteration % args.eval_freq == 1:
            model_dir = Path(args.save_path) / run_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)

        # === Learning Rate Annealing ===
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # === Rollout (采集数据) ===
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # 存储 Obs: DictArray 会自动处理所有 key
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step Env
            next_obs, reward, done, infos = env.step(action_pre_process_fn(action))
            next_obs = post_process_fn(next_obs)

            reward = torch.as_tensor(reward, device=device)
            terminations = torch.as_tensor(done, device=device)
            # 处理 Truncation (Time limit)
            if isinstance(infos, dict) and "truncation" in infos:
                truncations = torch.as_tensor(infos["truncation"], device=device)
            else:
                truncations = torch.zeros_like(terminations)

            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale
            running_episode_rewards += reward.view(-1)
            running_episode_lengths += 1

            if next_done.any():
                done_indices = torch.where(next_done)[0]
                avg_ep_reward = running_episode_rewards[done_indices].mean().item()
                avg_ep_len = running_episode_lengths[done_indices].mean().item()
                if logger:
                    logger.add_scalar("train/episode_reward", avg_ep_reward, global_step)
                    logger.add_scalar("train/episode_length", avg_ep_len, global_step)
                running_episode_rewards[done_indices] = 0
                running_episode_lengths[done_indices] = 0

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    if logger: logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time

        # === GAE Calculation (计算优势函数) ===
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]

                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # === PPO Update (优化更新) ===
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl: break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef,
                                                                args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl: break

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # === Logs ===
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if logger:
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)

    # Save final model
    if args.save_model and not args.evaluate:
        model_path = Path(args.save_path) / run_name / "final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    env.close()
    if logger: logger.close()