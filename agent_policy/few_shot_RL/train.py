import numpy as np
import torch
import argparse
import os
import time
import json
import policy_utils
import clip

from data_augs import center_crop
from logger import Logger
from video import VideoRecorder

from sac_new import (
    RadSacAgent,
    E2CSacAgent,
    DINOE2CSacAgent,
    DINOOnlySacAgent,
    E2CILQRAgent,
)

import env_wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--domain_name", default="robosuite_lift")
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--pre_transform_image_size", default=128, type=int)
    parser.add_argument("--cameras", nargs="+", default=[0, 1], type=int)
    parser.add_argument("--observation_type", default="pixel")
    parser.add_argument("--reward_type", default="dense")
    parser.add_argument("--control", default=None)

    parser.add_argument("--image_size", default=112, type=int)
    parser.add_argument("--action_repeat", default=1, type=int)
    parser.add_argument("--frame_stack", default=1, type=int)
    parser.add_argument("--num_updates", default=1, type=int)
    # replay buffer
    parser.add_argument("--replay_buffer_capacity", default=100000, type=int)
    parser.add_argument("--replay_buffer_load_dir", default="/data1/user/zhangshaolong/IL_RL/LaNE/demo/robosuite_lift/10", type=str)
    parser.add_argument(
        "--replay_buffer_keep_loaded", default=True, action="store_true"
    )
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--model_step", default=40000, type=str)
    # train
    parser.add_argument("--agent", default="dino_e2c_sac", type=str)
    parser.add_argument("--init_steps", default=0, type=int)
    parser.add_argument("--num_train_steps", default=10000, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)
    # eval
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--num_eval_episodes", default=2, type=int)
    # critic
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)
    # actor
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)
    # encoder
    parser.add_argument("--encoder_type", default="pixel", type=str)
    parser.add_argument("--encoder_feature_dim", default=32, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--latent_dim", default=128, type=int)
    # sac
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)
    # misc
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--work_dir", default="./data/robosuite_lift/ablation_rad", type=str)
    parser.add_argument("--save_tb", default=True, action="store_true")
    parser.add_argument("--save_buffer", default=True, action="store_true")
    parser.add_argument("--save_video", default=True, action="store_true")
    parser.add_argument("--save_sac", default=True, action="store_true")
    parser.add_argument("--detach_encoder", default=False, action="store_true")
    # Regularization
    parser.add_argument("--v_clip_low", default=None, type=float)
    parser.add_argument("--v_clip_high", default=None, type=float)
    parser.add_argument("--action_noise", default=None, type=float)
    # Final density of demos sampled from replay buffer
    parser.add_argument("--final_demo_density", default=None, type=float)

    parser.add_argument("--data_augs", default="crop", type=str)
    parser.add_argument("--log_interval", default=200, type=int)
    parser.add_argument("--pretrain_mode", default=None, type=str)
    parser.add_argument("--conv_layer_norm", default=True, action="store_true")
    parser.add_argument("--p_reward", default=1, type=float)

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, task_text, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = "stochastic_" if sample_stochastically else ""
        num_successes = 0
        for i in range(num_episodes):
            obs = env.reset()
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
                with policy_utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs, task_text)
                    else:
                        action = agent.select_action(obs,task_text)
                obs, reward, done, info = env.step(action)
                if info.get("is_success") or reward > 0:
                    episode_success = True
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
        key = args.domain_name + "-" + str(args.task_name) + "-" + args.data_augs
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
    elif args.agent == "e2c_sac":
        agent_class = E2CSacAgent
    elif args.agent == "dino_e2c_sac":
        agent_class = DINOE2CSacAgent
    elif args.agent == "dino_only_sac":
        agent_class = DINOOnlySacAgent
    elif args.agent == "e2c_ilqr":
        agent_class = E2CILQRAgent
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


def main():
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    exp_id = str(int(np.random.random() * 100000))
    policy_utils.set_seed_everywhere(args.seed)

    env = env_wrapper.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        from_pixels=args.observation_type == "pixel",
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        cameras=args.cameras,
    )

    test_env = env_wrapper.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        from_pixels=args.observation_type == "pixel",
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        cameras=args.cameras,
    )

    if args.encoder_type == "pixel" or args.encoder_type == "dino":
        env = policy_utils.FrameStack(env, k=args.frame_stack)
        test_env = policy_utils.FrameStack(test_env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    if args.task_name is None:
        env_name = args.domain_name
    else:
        env_name = args.domain_name + "-" + args.task_name
    exp_name = (
        args.reward_type
        + "-"
        + args.agent
        + "-"
        + args.encoder_type
        + "-"
        + args.data_augs
    )
    exp_name += (
        "-"
        + ts
        + "-"
        + env_name
        + "-im"
        + str(args.image_size)
        + "-b"
        + str(args.batch_size)
        + "-nu"
        + str(args.num_updates)
    )

    exp_name += "-s" + str(args.seed)

    exp_name += "-id" + exp_id
    args.work_dir = args.work_dir + "/" + exp_name
    policy_utils.make_dir(args.work_dir)
    video_dir = policy_utils.make_dir(os.path.join(args.work_dir, "video"))
    model_dir = policy_utils.make_dir(os.path.join(args.work_dir, "model"))
    buffer_dir = policy_utils.make_dir(os.path.join(args.work_dir, "buffer"))

    print("Working in directory:", args.work_dir)

    video = VideoRecorder(
        video_dir if args.save_video else None, camera_id=args.cameras[0]
    )

    with open(os.path.join(args.work_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    task_text = ["lift"]
    task_text = clip.tokenize([task_text[0]]).to(device)

    action_shape = env.action_space.shape

    if args.encoder_type == "pixel" or args.encoder_type == "dino":
        cpf = 3 * len(args.cameras)
        obs_shape = (cpf * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (
            cpf * args.frame_stack,
            args.pre_transform_image_size,
            args.pre_transform_image_size,
        )
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = policy_utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        load_dir=args.replay_buffer_load_dir,
        keep_loaded=args.replay_buffer_keep_loaded,
    )

    print("Starting with replay buffer filled to {}.".format(replay_buffer.idx))

    agent = make_agent(
        obs_shape=obs_shape, action_shape=action_shape, args=args, device=device
    )
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
        evaluate(test_env, agent, video, args.num_eval_episodes, L, step, task_text, args)

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
            time_acting += time.time() - time_start
            episode_reward = 0
            episode_step = 0
            episode += 1
            L.log("train/episode", episode, step)

        # sample action for data collection
        if step < 0:
            action = env.action_space.sample()
        else:
            with policy_utils.eval_mode(agent):
                action = agent.sample_action(obs, task_text)

        # run training update
        time_start = time.time()

        if step >= args.init_steps:
            for nu in range(args.num_updates):
                if args.final_demo_density is not None:
                    demo_density = args.final_demo_density
                else:
                    demo_density = None
                agent.update(replay_buffer, L, step, demo_density=demo_density, task_text=task_text)

        time_computing += time.time() - time_start

        time_start = time.time()

        next_obs, reward, done, _ = env.step(action)
        time_acting += time.time() - time_start

        # allow infinite bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1
        step += 1

    step = args.num_train_steps
    print("time spent computing:", time_computing)
    print("time spent acting:", time_acting)
    eval_and_save()
    env.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
