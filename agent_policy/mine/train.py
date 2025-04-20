import numpy as np
import torch
import argparse
import os
import time
import json
from omegaconf import OmegaConf
import agent_policy.mine.utils as utils
import agent_policy.mine.env_wrapper as env_wrapper
from agent_policy.mine.data_augs import center_crop
from agent_policy.mine.logger import Logger
from agent_policy.mine.video import VideoRecorder
from agent_policy.mine.sac import (
    RadSacAgent,
    MineSacAgent,
)

def evaluate(env, agent, video, num_episodes, L, step, args, post_process_fn=None, reward_fn=None, action_pre_process_fn=None):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = "stochastic_" if sample_stochastically else ""
        num_successes = 0
        for i in range(num_episodes):
            obs = env.reset()
            if post_process_fn is not None:
                obs = post_process_fn(obs)
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            episode_success = False
            while not done:
                # center crop image
                if (
                    args.encoder_type == "pixel"
                    or "crop" in args.data_augs
                    or "translate" in args.data_augs
                ):
                    if isinstance(obs, list):
                        obs[0] = center_crop(obs[0], args.image_size)
                    else:
                        obs = center_crop(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                real_action = action_pre_process_fn(action, agent.replay_buffer.xyz_mean[0], agent.replay_buffer.xyz_std[0])
                # pre_reward, _, pre_reward_info = reward_fn(obs, real_action, None, -1, reward_type="test", is_save=False, is_train=False)
                obs, reward, done, info = env.step(real_action)
                post_reward, done, post_reward_info = reward_fn(obs, real_action, info, reward, reward_type=args.reward_model_type, is_save=False, is_train=False)
                if post_process_fn is not None:
                    obs = post_process_fn(obs)
                if info['is_success'] == True:
                    episode_success = True
                if info["truncation"] == True:
                    done = True
                video.record(env)
                episode_reward += reward
            num_successes += episode_success

            video.save("%d.mp4" % step)
            L.log("eval/" + prefix + "episode_reward", episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log("eval/" + prefix + "eval_time", time.time() - start_time, step)
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
        L.log("eval/" + prefix + "mean_episode_reward", mean_ep_reward, step)
        L.log("eval/" + prefix + "best_episode_reward", best_ep_reward, step)
        L.log("eval/" + prefix + "success_rate", success_rate, step)

        filename = args.work_dir + "/eval_scores.npy"
        key = str(args.task_name) + "-" + args.data_augs
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
        log_data[key][step]["env_step"] = step * args.action_repeat

        np.save(filename, log_data)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == "rad_sac":
        agent_class = RadSacAgent
    elif args.agent == "dino_e2c_sac":
        agent_class = MineSacAgent
    else:
        agent_class = None
    return agent_class(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_type=args.encoder_type,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_tau=args.encoder_tau,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        log_interval=args.log_interval,
        detach_encoder=args.detach_encoder,
        latent_dim=args.latent_dim,
        v_clip_low=args.v_clip_low,
        v_clip_high=args.v_clip_high,
        action_noise=args.action_noise,
        conv_layer_norm=args.conv_layer_norm,
        data_augs=args.data_augs,
        p_reward=args.p_reward,
    )


def mine_train_main(agent, obs_shape, args, env, test_env, post_process_fn=None, reward_fn=None, action_pre_process_fn=None):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    # utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, "video"))
    model_dir = utils.make_dir(os.path.join(args.work_dir, "model"))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, "buffer"))

    print("Working in directory:", args.work_dir)

    video = VideoRecorder(
        video_dir if args.save_video else None, camera_id=args.cameras[0]
    )

    with open(os.path.join(args.work_dir, "args.json"), "w") as f:
        json.dump(OmegaConf.to_container(args, resolve=True), f, sort_keys=True, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_shape = env.action_space.shape

    if args.encoder_type == "pixel" or args.encoder_type == "dino":
        cpf = 3 * len(args.cameras)
        pre_aug_obs_shape = (
            cpf * args.frame_stack,
            args.pre_transform_image_size,
            args.pre_transform_image_size,
        )
    else:
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        load_dir=args.replay_buffer_load_dir,
        keep_loaded=args.replay_buffer_keep_loaded,
    )

    agent = make_agent(
        obs_shape=obs_shape, action_shape=action_shape, args=args, device=device
    )

    print("Starting with replay buffer filled to {}.".format(replay_buffer.idx))

    agent.replay_buffer = replay_buffer
    if args.model_dir is not None:
        agent.load(args.model_dir, args.model_step)
    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    def eval_and_save():
        if args.save_buffer:
            replay_buffer.save(buffer_dir)
        if args.save_sac:
            agent.save(model_dir, step)
        L.log("eval/episode", episode, step)
        print("evaluating")
        evaluate(test_env, agent, video, args.num_eval_episodes, L, step, args, post_process_fn,  reward_fn, action_pre_process_fn)

    time_computing = 0
    time_acting = 0
    step = 0

    while step < args.num_train_steps:
        # evaluate agent periodically
        if step % args.eval_freq == 0:
            eval_and_save()

        if done:
            if step > 0:
                L.log("train/duration", time.time() - start_time, step)
                L.dump(step)
                start_time = time.time()
            L.log("train/episode_reward", episode_reward, step)

            time_start = time.time()
            obs = env.reset()
            if post_process_fn is not None:
                obs = post_process_fn(obs)
            time_acting += time.time() - time_start
            episode_reward = 0
            episode_step = 0
            episode += 1
            L.log("train/episode", episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        time_start = time.time()

        if step >= args.init_steps:
            for nu in range(args.num_updates):
                if args.final_demo_density is not None:
                    demo_density = args.final_demo_density
                else:
                    demo_density = None
                agent.update(replay_buffer, L, step, demo_density=demo_density)

        time_computing += time.time() - time_start

        time_start = time.time()
        real_action = action_pre_process_fn(action, agent.replay_buffer.xyz_mean[0], agent.replay_buffer.xyz_std[0])
        print("real_action: ", real_action)
        next_obs, reward, done, info = env.step(real_action)
        if reward_fn is not None:
            reward, done, _ = reward_fn(next_obs, real_action, info, reward, reward_type=args.reward_model_type, is_save=args.save_online_image, is_train=False)
        if post_process_fn is not None:
            next_obs = post_process_fn(next_obs)
        time_acting += time.time() - time_start

        # allow infinite bootstrap
        done_bool = 0 if episode_step + 1 == env.all_task_max_num else float(done)
        if info["truncation"] == True:
            done = True
        episode_reward += reward
        print("step: ", step, "episode_reward: ", episode_reward, "episode_step: ", episode_step, "done: ", done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1
        step += 1

    step = args.num_train_steps
    print("time spent computing:", time_computing)
    print("time spent acting:", time_acting)
    eval_and_save()
    env.close()
