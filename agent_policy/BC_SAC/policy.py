import torch
import torch.nn as nn
import torch.nn.functional as F
from sac_agent import SacAgent

def train_IL_policy(agent):
    for epoch in range(starting_epoch, config.num_epochs):
        for step_idx in range(config.steps_per_epoch):
            step = epoch * config.steps_per_epoch + step_idx
            batch = random.sample(dataset, config.batch_size)
            batch_state = torch.stack([b["state"] for b in batch]).to(config.device)
            gt_action = torch.stack([b["action"] for b in batch]).to(config.device)
            gt_q = torch.stack([b["q_target"] for b in batch]).to(config.device)

            # Train policy with BC.
            agent.pi_optimizer.zero_grad()
            pred_action, pi, log_pi, log_std = agent.actor(obs, detach_encoder=True)
            actor_loss = ((pred_action - gt_action)**2).mean()
            actor_loss.backward()
            agent.actor_optimizer.step()
            agent.log_alpha_optimizer.zero_grad()
            alpha_loss = (agent.alpha * (-log_pi - agent.target_entropy).detach()).mean()
            alpha_loss.backward()
            agent.log_alpha_optimizer.step()
            # get current Q estimates
            current_Q1, current_Q2 = agent.critic(
                obs, pred_action, detach_encoder=agent.detach_encoder
            )
            critic_loss = F.mse_loss(current_Q1, gt_q) + F.mse_loss(
                current_Q2, gt_q
            )
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

def train_RL_policy(agent):
    pass

if __name__ == "__main__":
    pass       