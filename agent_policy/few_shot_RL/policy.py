import numpy as np
import torch
import argparse
import os
import time
import json
import clip
import agent_policy.few_shot_RL.policy_utils
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

class FewDemoPolicy:
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

    def get_action(self, obs, task_text):
        action = self.agent.sample_action(obs, task_text)

    def train(self):
        IL_agent_name = "dino_e2c_sac"
        RL_agent_name = "dino_only_sac"
        torch.multiprocessing.set_start_method("spawn")
        self.train_IL_policy(IL_agent_name=IL_agent_name)
        # self.train_RL_policy(RL_agent_name=RL_agent_name)

    def train_IL_policy(self, IL_agent_name):
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
            obs_shape = self.env.observation_space.shape
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
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        time_computing = 0
        time_acting = 0
        step = 0
        task_text = ["lift"]
        task_text = clip.tokenize([task_text[0]]).to(self.device)
        while step < self.params['num_train_steps']:
            # evaluate agent periodically
            if step % self.params['eval_freq'] == 0:
                if self.params['save_buffer']:
                    replay_buffer.save(self.buffer_dir)
                if self.params['save_sac']:
                    agent.save(self.model_dir, step)
                self.L.log("eval/episode", episode, step)
                self.eval_policy(agent, self.params['num_eval_episodes'], step, task_text)
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
            if step < 0:
                action = self.env.action_space.sample()
            else:
                with policy_utils.eval_mode(agent):
                    action = agent.sample_action(obs, task_text)

            # run training update
            time_start = time.time()

            if step >= self.params['init_steps']:
                for nu in range(self.params['num_updates']):
                    if self.params['final_demo_density'] is not None:
                        demo_density = self.params['final_demo_density']
                    else:
                        demo_density = None
                    agent.update(replay_buffer, self.L, step, demo_density=demo_density, task_text=task_text)

            time_computing += time.time() - time_start

            time_start = time.time()

            next_obs, reward, done, _ = self.env.step(action)
            time_acting += time.time() - time_start

            # allow infinite bootstrap
            done_bool = 0 if episode_step + 1 == self.env ._max_episode_steps else float(done)
            episode_reward += reward

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
        self.eval_policy(agent, self.params['num_eval_episodes'], step, task_text)
        print("evaluating")
        self.env.close()

    def train_RL_policy(self, RL_agent_name):
        pass

    def eval_policy(self, agent, num_episodes, step, task_text, sample_stochastically=False):
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
            while not done:
                if isinstance(obs, list):
                    obs[0] = center_crop(obs[0], self.params['image_size'])
                else:
                    obs = center_crop(obs, self.params['image_size'])
                with policy_utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs, task_text)
                    else:
                        action = agent.select_action(obs,task_text)
                obs, reward, done, info = self.env.step(action)
                if info.get("is_success") or reward > 0:
                    episode_success = True
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
        key = self.params['domain_name'] + "-" + str(self.params['task_name']) + "-" + self.params['data_augs']
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

if __name__ == "__main__":
    params = {
    "task_name": "robosuite_lift",
    "pre_transform_image_size": 128,
    "cameras": [0, 1],
    "observation_type": "pixel",
    "reward_type": "dense",
    "control": None,
    "image_size": 112,
    "action_repeat": 1,
    "frame_stack": 1,
    "num_updates": 1,
    "replay_buffer_capacity": 100000,
    "replay_buffer_load_dir": "/data1/user/zhangshaolong/IL_RL/LaNE/demo/robosuite_lift/10",
    "replay_buffer_keep_loaded": True,
    "model_dir": None,
    "model_step": 40000,
    "agent_name": "dino_e2c_sac",
    "init_steps": 0,
    "num_train_steps": 10000,
    "batch_size": 128,
    "hidden_dim": 1024,
    "eval_freq": 1000,
    "num_eval_episodes": 2,
    "critic_lr": 1e-3,
    "critic_beta": 0.9,
    "critic_tau": 0.01,
    "critic_target_update_freq": 2,
    "actor_lr": 1e-3,
    "actor_beta": 0.9,
    "actor_log_std_min": -10,
    "actor_log_std_max": 2,
    "actor_update_freq": 2,
    "encoder_type": "pixel",
    "encoder_feature_dim": 32,
    "encoder_tau": 0.05,
    "num_layers": 4,
    "num_filters": 32,
    "latent_dim": 128,
    "discount": 0.99,
    "init_temperature": 0.1,
    "alpha_lr": 1e-4,
    "alpha_beta": 0.5,
    "seed": 1,
    "work_dir": "./data/robosuite_lift/ablation_rad",
    "save_tb": True,
    "save_buffer": True,
    "save_video": True,
    "save_sac": True,
    "detach_encoder": False,
    "v_clip_low": None,
    "v_clip_high": None,
    "action_noise": None,
    "final_demo_density": None,
    "data_augs": "crop",
    "log_interval": 200,
    "pretrain_mode": None,
    "conv_layer_norm": True,
    "p_reward": 1,
}
    