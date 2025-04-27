# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
from . import utils as maniwhere_utils
from .logger import Logger
from .replay_buffer import ReplayBufferStorage, make_replay_loader
from .video import TrainVideoRecorder, VideoRecorder
import wandb
import time
import gc
from typing import List
import imageio
from collections import deque

torch.backends.cudnn.benchmark = True
from .algos.maniwhere import ManiAgent
from .utils import CameraViewWrapper


def make_agent(obs_shape, action_shape, cfg):
    return ManiAgent(obs_shape = obs_shape, action_shape = action_shape,
               device = cfg.device, lr = cfg.lr, feature_dim = cfg.feature_dim,
                 hidden_dim = cfg.hidden_dim, critic_target_tau = cfg.critic_target_tau, num_expl_steps = cfg.num_expl_steps,
                 update_every_steps = cfg.update_every_steps, stddev_schedule = 'linear(1.0,0.1,600000)', stddev_clip = cfg.stddev_clip, use_tb = False, use_wandb = False,
                 temp = cfg.temp, aux_coef = cfg.aux_coef, aux_l2_coef = cfg.aux_l2_coef, aux_tcc_coef = cfg.aux_tcc_coef, aux_latency = cfg.aux_latency, lr_stn = cfg.lr_stn)

class Workspace:
    def __init__(self, cfg, train_env, test_env, obs_shape, action_shape):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.train_env = train_env
        self.eval_env = test_env
        self.cfg = cfg
        maniwhere_utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        extra_channel = 1 if cfg.use_depth else 0
        num_frames = cfg.frame_stack
        self.observation_spec = specs.BoundedArray(
            shape=np.array([3 * num_frames + extra_channel, cfg.img_size, cfg.img_size]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        self.action_spec = specs.BoundedArray(action_shape,
                                               dtype = np.float32,
                                               minimum=-cfg.env_info.max_action,
                                               maximum=cfg.env_info.max_action,
                                               name='action')
        self.setup()
        self.agent = make_agent(self.observation_spec.shape,
                                self.action_spec.shape,
                                self.cfg.agent)
        self.timer = maniwhere_utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._obs_channel = self.observation_spec.shape[0]
        self.best_eval_reward = 0

    def setup(self):
        if self.cfg.use_wandb:
            exp_name = '_'.join([
                self.cfg.task_name,
                str(self.cfg.seed)
            ])
            wandb.init(project="sim2real", group=self.cfg.wandb_group, name=exp_name)
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb)
        # create replay buffer
        data_specs = (self.observation_spec,
                      self.action_spec,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      specs.Array((1,), np.float32, 'not_done'),)

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None
        self.stored_episodes = deque([], maxlen=20) if self.cfg.use_traj else None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self, post_process_fn = None, reward_fn = None, action_pre_process_fn = None):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = maniwhere_utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            obs = self.eval_env.reset()
            time_step = post_process_fn(obs, reward = 0.0, info = {"truncation": False}, action = np.zeros(self.action_spec.shape), is_reset=True, is_train = False)
            self.video_recorder.init(obs["sceneview_image"], enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), maniwhere_utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                action = action_pre_process_fn(action, action_mean = reward_fn.replay_buffer.xyz_mean, action_std = reward_fn.replay_buffer.xyz_std)
                obs, reward, done, info = self.eval_env.step(action)
                time_step = post_process_fn(obs, reward = reward, info = info, action = action, is_reset=False, is_train = False)
                self.video_recorder.record(obs["sceneview_image"])
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
        
        if (self.best_eval_reward < (total_reward / episode)) and self.cfg.save_snapshot and self.global_step >= int(5e5):
            self.best_eval_reward = (total_reward / episode)
            self.save_snapshot(best=True, step=self.global_step)
            print('final period best eval reward:', self.best_eval_reward)
        

    def train(self, post_process_fn = None, reward_fn = None, action_pre_process_fn = None):
        # predicates
        train_until_step = maniwhere_utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = maniwhere_utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = maniwhere_utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        # episodic_list is used to store the observation of each episode
        episodic_list: List[np.ndarray] = []

        obs = self.train_env.reset()
        time_step = post_process_fn(obs, reward = 0.0, info = {"truncation": False}, action = np.zeros(self.action_spec.shape), is_reset=True, is_train = True)
        episodic_list.append(time_step.observation[self._obs_channel:].copy())
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                if self.cfg.use_traj:
                    self.stored_episodes.append(episodic_list)
                episodic_list = []
                # reset env
                obs = self.train_env.reset()
                time_step = post_process_fn(obs, reward = 0.0, info = {"truncation": False}, action = np.zeros(self.action_spec.shape), is_reset=True, is_train = True)
                episodic_list.append(time_step.observation[self._obs_channel:].copy())
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot and (self.global_step % int(2e4) == 0):
                    self.save_snapshot(step=self.global_step)
                episode_step = 0
                episode_reward = 0
                
                # aux_lr_scheduler
                # self.agent.aux_opt_scheduler.step()
                # self.agent.stn_opt_scheduler.step()

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval(post_process_fn, reward_fn, action_pre_process_fn)

            # sample action
            with torch.no_grad(), maniwhere_utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation[:self._obs_channel],
                                        self.global_step,
                                        eval_mode=False)

            

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.stored_episodes, self.global_step, reward_model_fn=reward_fn)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            action = action_pre_process_fn(action, action_mean = reward_fn.replay_buffer.xyz_mean, action_std = reward_fn.replay_buffer.xyz_std)
            obs, reward, done, info = self.train_env.step(action)
            time_step = post_process_fn(obs, reward = reward, info = info, action = action, is_reset=False, is_train = True)
            episodic_list.append(time_step.observation[self._obs_channel:].copy())
            
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, best=False, step=None):
        if best:
            snapshot = self.work_dir / f'best_snapshot_{step}.pt'
        else:
            snapshot = self.work_dir / f'snapshot_{step}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

def maniwhere_train_main(agent, args, obs_shape, action_shape, env, test_env, post_process_fn=None, reward_fn=None, action_pre_process_fn=None):
    root_dir = Path.cwd()
    workspace = Workspace(args, env, test_env, obs_shape, action_shape)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train(post_process_fn, reward_fn, action_pre_process_fn)
