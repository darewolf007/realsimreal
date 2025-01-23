import numpy as np
import torch
import argparse
import os
import time
import json
import clip
import torch.nn.functional as F
import agent_policy.few_shot_RL.policy_utils as policy_utils
from agent_policy.few_shot_RL.logger import Logger
from agent_policy.few_shot_RL.data_augs import center_crop
from agent_policy.few_shot_RL.video import VideoRecorder
from agent_policy.few_shot_RL.sac_new_single import (
    RadSacAgent,
    E2CSacAgent,
    DINOE2CSacAgent,
    DINOOnlySacAgent,
    E2CILQRAgent,
)

def get_latest_model_step(model_dir):
    files = os.listdir(model_dir)
    steps = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith("actor_") and f.endswith(".pt")]
    if not steps:
        raise ValueError(f"No valid actor_*.pt files found in {model_dir}")
    return max(steps)

class BCSACPolicy:
    def __init__(
        self,
        env,
        device,
        params,
    ):
        self.device = device
        self.params = params
        self.env = env
        self.L = Logger(self.params['work_dir'], use_tb=self.params['save_tb'])
        video_dir = policy_utils.make_dir(os.path.join(self.params['work_dir'], "video"))
        self.model_dir = policy_utils.make_dir(os.path.join(self.params['work_dir'], "model"))
        self.buffer_dir = policy_utils.make_dir(os.path.join(self.params['work_dir'], "buffer"))
        self.video = VideoRecorder(video_dir if self.params['save_video'] else None, camera_id=self.params['cameras'][0])
        self.init_tokenize()
    
    def init_agent(self, agent_class, obs_shape, action_shape):
        agent = agent_class(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self.device,
            hidden_dim=self.params["hidden_dim"],
            discount=self.params["discount"],
            init_temperature=self.params["init_temperature"],
            alpha_lr=self.params["alpha_lr"],
            alpha_beta=self.params["alpha_beta"],
            actor_lr=self.params["actor_lr"],
            actor_beta=self.params["actor_beta"],
            actor_log_std_min=self.params["actor_log_std_min"],
            actor_log_std_max=self.params["actor_log_std_max"],
            actor_update_freq=self.params["actor_update_freq"],
            critic_lr=self.params["critic_lr"],
            critic_beta=self.params["critic_beta"],
            critic_tau=self.params["critic_tau"],
            critic_target_update_freq=self.params["critic_target_update_freq"],
            encoder_type=self.params["encoder_type"],
            encoder_feature_dim=self.params["encoder_feature_dim"],
            encoder_tau=self.params["encoder_tau"],
            num_layers=self.params["num_layers"],
            num_filters=self.params["num_filters"],
            log_interval=self.params["log_interval"],
            detach_encoder=self.params["detach_encoder"],
            latent_dim=self.params["latent_dim"],
            v_clip_low=self.params["v_clip_low"],
            v_clip_high=self.params["v_clip_high"],
            action_noise=self.params["action_noise"],
            conv_layer_norm=self.params["conv_layer_norm"],
            data_augs=self.params["data_augs"],
            p_reward=self.params["p_reward"],
        )
        return agent

    def choose_agent(self, agent_name):
        if agent_name == "rad_sac":
            agent_class = RadSacAgent
        elif agent_name == "e2c_sac":
            agent_class = E2CSacAgent
        elif agent_name == "dino_e2c_sac":
            agent_class = DINOE2CSacAgent
        elif agent_name == "dino_only_sac":
            agent_class = DINOOnlySacAgent
        elif agent_name == "e2c_ilqr":
            agent_class = E2CILQRAgent
        else:
            raise NotImplementedError
        return agent_class

    def get_action(self, obs, task_text_token):
        action = self.agent.sample_action(obs, task_text_token)

    def train(self):
        IL_agent_name = "rad_sac"
        RL_agent_name = "rad_sac"
        torch.multiprocessing.set_start_method("spawn")
        # self.train_BC_policy(IL_agent_name=IL_agent_name)
        self.train_RL_policy(RL_agent_name=RL_agent_name)

    def init_tokenize(self):
        self.subtask_promot_tokens = []
        self.approaching_token = clip.tokenize("approaching").to(self.device)
        for sub_task in self.params['sub_task_promot']:
            task_text_token = clip.tokenize(sub_task).to(self.device)
            self.subtask_promot_tokens.append(task_text_token)

    def get_text(self, subtask_id):
        if subtask_id == -1:
            task_text_tokens = self.approaching_token
        else:
            task_text_tokens = self.subtask_promot_tokens[subtask_id]
        return task_text_tokens

    def train_RL_policy(self, RL_agent_name):
        policy_utils.set_seed_everywhere(self.params['seed'])
        action_shape = self.env.action_space.shape
        if self.params['encoder_type'] == "pixel" or self.params['encoder_type'] == "dino":
            cpf = 3 * len(self.params['cameras'])
            obs_shape = (cpf * self.params['frame_stack'], self.params['image_size'], self.params['image_size'])
            pre_aug_obs_shape = (
                cpf * self.params['frame_stack'],
                self.params['pre_transform_image_size'],
                self.params['pre_transform_image_size'],
            )
        else:
            obs_shape = self.env.observation_shape
            pre_aug_obs_shape = obs_shape

        replay_buffer = policy_utils.ReplayBuffer(
            obs_shape=pre_aug_obs_shape,
            action_shape=action_shape,
            capacity=self.params['replay_buffer_capacity'],
            batch_size=self.params['batch_size'],
            device=self.device,
            image_size=self.params['image_size'],
            load_dir=self.params['replay_buffer_load_dir'],
            keep_loaded=self.params['replay_buffer_keep_loaded'],
        )
        agent = self.init_agent(self.choose_agent(RL_agent_name), obs_shape, action_shape)
        agent.replay_buffer = replay_buffer
        model_dir = self.model_dir
        model_step = get_latest_model_step(model_dir)
        agent.load(model_dir, model_step)
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        time_computing = 0
        time_acting = 0
        step = 0
        task_text_token = self.subtask_promot_tokens[0]
        while step < self.params['num_train_steps']:
            task_text_token = None
            # evaluate agent periodically
            if step % self.params['eval_freq'] == 0:
                if self.params['save_buffer']:
                    replay_buffer.save(self.buffer_dir)
                if self.params['save_sac']:
                    agent.save(self.model_dir, step)
                self.L.log("eval/episode", episode, step)
                self.eval_policy(agent, self.params['num_eval_episodes'], step)
                print("evaluating")

            if done:
                if step > 0:
                    self.L.log("train/duration", time.time() - start_time, step)
                    self.L.dump(step)
                    start_time = time.time()
                self.L.log("train/episode_reward", episode_reward, step)

                time_start = time.time()
                obs = self.env.reset()
                time_acting += time.time() - time_start
                episode_reward = 0
                episode_step = 0
                episode += 1
                self.L.log("train/episode", episode, step)

            # sample action for data collection
            if self.params['init_steps'] is not None:
                if step < self.params['init_steps']:
                    with policy_utils.eval_mode(agent):
                        action = agent.sample_action(obs, task_text_token)
                        action[3:-1] = np.array([0,0,0])
                # else:
                #     action = self.env.action_space.sample()
                #     action[3:-1] = np.array([0,0,0])
            else:
                with policy_utils.eval_mode(agent):
                    action = agent.sample_action(obs, task_text_token)
                    action[3:-1] = np.array([0,0,0])

            # run training update
            time_start = time.time()

            if step >= self.params['init_steps']:
                for nu in range(self.params['num_updates']):
                    if self.params['final_demo_density'] is not None:
                        demo_density = self.params['final_demo_density']
                    else:
                        demo_density = None
                    agent.update(replay_buffer, self.L, step, demo_density=demo_density, task_text=task_text_token)

            time_computing += time.time() - time_start

            time_start = time.time()

            next_obs, reward, done, info = self.env.step(action)
            task_text_token = self.subtask_promot_tokens[self.env.sub_task_idx]
            time_acting += time.time() - time_start

            # allow infinite bootstrap
            done_bool = 0 if episode_step + 1 == self.env.all_task_max_num else float(done)
            if episode_step + 1 == self.env.all_task_max_num:
                done = True
            episode_reward += reward
            print("step:", step, "episode:", episode, "episode_step:", episode_step, "reward:", reward, "done:", done)
            replay_buffer.add(obs, action, reward, next_obs, done_bool)

            obs = next_obs
            episode_step += 1
            step += 1


        step = self.params['num_train_steps']
        print("time spent computing:", time_computing)
        print("time spent acting:", time_acting)
        if self.params['save_buffer']:
            replay_buffer.save(self.buffer_dir)
        if self.params['save_sac']:
            agent.save(self.model_dir, step)
        self.L.log("eval/episode", episode, step)
        self.eval_policy(agent, self.params['num_eval_episodes'], step)
        print("evaluating")
        self.env.close()

    def train_BC_policy(self, IL_agent_name):
        policy_utils.set_seed_everywhere(self.params['seed'])
        action_shape = self.env.action_space.shape
        if self.params['encoder_type'] == "pixel" or self.params['encoder_type'] == "dino":
            cpf = 3 * len(self.params['cameras'])
            obs_shape = (cpf * self.params['frame_stack'], self.params['image_size'], self.params['image_size'])
            pre_aug_obs_shape = (
                cpf * self.params['frame_stack'],
                self.params['pre_transform_image_size'],
                self.params['pre_transform_image_size'],
            )
        else:
            obs_shape = self.env.observation_shape
            pre_aug_obs_shape = obs_shape

        replay_buffer = policy_utils.ReplayBuffer(
            obs_shape=pre_aug_obs_shape,
            action_shape=action_shape,
            capacity=self.params['replay_buffer_capacity'],
            batch_size=self.params['batch_size'],
            device=self.device,
            image_size=self.params['image_size'],
            load_dir=self.params['replay_buffer_load_dir'],
            keep_loaded=self.params['replay_buffer_keep_loaded'],
        )
        agent = self.init_agent(self.choose_agent(IL_agent_name), obs_shape, action_shape)
        agent.replay_buffer = replay_buffer
        if self.params['model_dir'] is not None:
            agent.load(self.params['model_dir'], self.params['model_step'])

        best_reward = -np.inf
        task_text_token = self.subtask_promot_tokens[0]
        for epoch in range(0, self.params['bc_train_steps']):
            print("evaluating")
            mean_ep_reward = self.eval_policy(agent, self.params['num_eval_episodes'], epoch)
            if mean_ep_reward >= best_reward:
                best_reward = mean_ep_reward
                agent.save(self.model_dir, epoch)
            for step_idx in range(2000):
                obs, gt_action, reward, next_obs, not_done = replay_buffer.sample_rad(agent.augs_funcs)
                # Train policy with BC.
                
                pred_action, pi, log_pi, log_std = agent.actor(obs, detach_encoder=True)
                # actor_loss = ((pred_action - gt_action)**2).mean() * 100
                pos_rot_loss = ((pred_action[:, :6] - gt_action[:, :6])**2).mean() * 100
                gripper_loss = ((pred_action[:, 6:] - gt_action[:, 6:])**2).mean() * 200
                actor_loss = pos_rot_loss + gripper_loss
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()
                agent.log_alpha_optimizer.zero_grad()
                alpha_loss = (agent.alpha * (-log_pi - agent.target_entropy).detach()).mean()
                alpha_loss.backward()
                agent.log_alpha_optimizer.step()
                # get current Q estimates
                with torch.no_grad():
                    _, policy_action, policy_log_pi, _ = agent.actor(next_obs)
                    target_Q1, target_Q2 = agent.critic_target(next_obs, policy_action)
                    if agent.v_clip_low is not None:
                        target_Q1 = target_Q1.clamp(agent.v_clip_low, agent.v_clip_high)
                        target_Q2 = target_Q2.clamp(agent.v_clip_low, agent.v_clip_high)
                    target_V = torch.min(target_Q1, target_Q2) - agent.alpha.detach() * policy_log_pi
                    target_Q = reward + (not_done * agent.discount * target_V)

                current_Q1, current_Q2 = agent.critic(
                    obs, gt_action, detach_encoder=agent.detach_encoder
                )
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                    current_Q2, target_Q
                )
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                agent.critic_optimizer.step()
                print("epoch:", epoch, "step_idx:", step_idx, "actor_loss:", actor_loss.item(), "critic_loss:", critic_loss.item(), "alpha_loss:", alpha_loss.item())
            policy_utils.soft_update_params(
                agent.critic.Q1, agent.critic_target.Q1, agent.critic_tau
            )
            policy_utils.soft_update_params(
                agent.critic.Q2, agent.critic_target.Q2, agent.critic_tau
            )
            policy_utils.soft_update_params(
                agent.critic.encoder, agent.critic_target.encoder, agent.encoder_tau
            )
        
    def eval_policy(self, agent, num_episodes, step, sample_stochastically=False):
        all_ep_rewards = []
        start_time = time.time()
        prefix = "stochastic_" if sample_stochastically else ""
        num_successes = 0
        for i in range(num_episodes):
            obs = self.env.reset()
            self.video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            episode_success = False
            task_text_token = self.subtask_promot_tokens[0]
            episode_step = 0
            while not done:
                task_text_token = None
                if isinstance(obs, list):
                    obs[0] = center_crop(obs[0], self.params['image_size'])
                else:
                    obs = center_crop(obs, self.params['image_size'])
                with policy_utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs, task_text_token)
                    else:
                        action = agent.select_action(obs,task_text_token)
                obs, reward, done, info = self.env.step(action)
                task_text_token = self.subtask_promot_tokens[self.env.sub_task_idx]
                if done:
                    episode_success = True
                    break
                if episode_step + 1 == self.env.all_task_max_num:
                    break
                episode_step += 1
                self.video.record(self.env)
                episode_reward += reward
            num_successes += episode_success
            self.video.save("%d.mp4" % step)
            self.L.log("eval/" + prefix + "episode_reward", episode_reward, step)
            all_ep_rewards.append(episode_reward)
        self.L.log("eval/" + prefix + "eval_time", time.time() - start_time, step)
        if num_episodes > 0:
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            std_ep_reward = np.std(all_ep_rewards)
            success_rate = num_successes / num_episodes
        else:
            mean_ep_reward = 0
            best_ep_reward = 0
            std_ep_reward = 0
            success_rate = 0
        self.L.log("eval/" + prefix + "mean_episode_reward", mean_ep_reward, step)
        self.L.log("eval/" + prefix + "best_episode_reward", best_ep_reward, step)
        self.L.log("eval/" + prefix + "success_rate", success_rate, step)
        filename = self.params['work_dir'] + "/eval_scores.npy"
        key = str(self.params['task_name']) + "-" + self.params['data_augs']
        try:
            log_data = np.load(filename, allow_pickle=True)
            log_data = log_data.item()
        except FileNotFoundError:
            log_data = {}
        if key not in log_data:
            log_data[key] = {}
        log_data[key][step] = {}
        log_data[key][step]["step"] = step
        log_data[key][step]["mean_ep_reward"] = mean_ep_reward
        log_data[key][step]["max_ep_reward"] = best_ep_reward
        log_data[key][step]["success_rate"] = success_rate
        log_data[key][step]["std_ep_reward"] = std_ep_reward
        log_data[key][step]["env_step"] = step * self.params['action_repeat']
        np.save(filename, log_data)
        self.L.dump(step)
        return mean_ep_reward
