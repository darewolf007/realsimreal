import torch
from agent_policy.LaNE.sac import DINOE2CSacAgent
from agent_policy.LaNE.train import LaNE_train_main
from agent_policy.LaNE.data_augs import center_crop
from agent_policy.BC_SAC.bc_sac_policy import BC_SAC_train_main
from agent_policy.BC_SAC.bc_sac_policy import BCSACPolicy

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

class BaseAgentPolicy:
    def __init__(self, args, agent_name, device, obs_shape, action_shape, is_train):
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.agent_name = agent_name
        self.args = args
        self.is_train = is_train
        if agent_name == "LaNE":
            self.agent = self.init_LaNE_agent(args)
        elif agent_name == "BC_SAC":
            self.agent = self.init_BC_SAC_agent(args)
        else:
            raise NotImplementedError
    
    def init_BC_SAC_agent(self, args):
        bc_sac = BCSACPolicy(env = None, device = "cuda", params = args)
        agent = bc_sac.init_agent(bc_sac.choose_agent("rad_sac"), self.obs_shape, self.action_shape)
        if self.is_train:
            print("Training agent")
        else:
            print("Loading agent")
            agent.load(args.model_dir, args.model_step)
        return agent

    def init_LaNE_agent(self, args):   
        agent = DINOE2CSacAgent(
        obs_shape=self.obs_shape,
        action_shape=self.action_shape,
        device=self.device,
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
        p_reward=args.p_reward,)
        if self.is_train:
            print("Training agent")
        else:
            print("Loading agent")
            agent.load(args.model_dir, 8000)
        return agent

    def get_action(self, obs):
        if self.agent_name == "LaNE":
            obs = center_crop(obs, self.args.image_size)
            with eval_mode(self.agent):
                action = self.agent.select_action(obs)
            return action
        else:
            raise NotImplementedError

    
    def train_agent(self, env, test_env, img_post_process_fn, reward_fn=None, action_pre_process_fn=None):
        if self.agent_name == "LaNE":
            torch.multiprocessing.set_start_method("spawn")
            LaNE_train_main(agent = self.agent, args = self.args, env = env, test_env = test_env, post_process_fn=img_post_process_fn, reward_fn=reward_fn, action_pre_process_fn=action_pre_process_fn)
        elif self.agent_name == "BC_SAC":
            torch.multiprocessing.set_start_method("spawn")
            BC_SAC_train_main(agent = None, args = self.args, env = env, test_env = test_env, post_process_fn=img_post_process_fn, reward_fn=reward_fn, action_pre_process_fn=action_pre_process_fn)
        else:
            raise NotImplementedError
