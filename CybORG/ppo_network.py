from ppo_utils import PPOActorCritic  # Import from the new file

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ppo_network import PPOActorCritic

class PPOAgent:
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.99, eps_clip=0.2, update_epochs=10):
        self.action_space = action_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs

        # Initialize Actor-Critic model and optimizer
        self.policy = PPOActorCritic(observation_space.shape[0], action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.policy.act(state)
        return action.item(), log_prob

    def update_policy(self, rollout_data, batch_size, update_epochs):
        states, actions, rewards, next_states, dones, old_log_probs = rollout_data
        dataset_size = len(states)
        
        for _ in range(update_epochs):
            for i in range(0, dataset_size, batch_size):
                batch_indices = slice(i, i + batch_size)
                self.update_step(
                    states[batch_indices], actions[batch_indices],
                    rewards[batch_indices], next_states[batch_indices],
                    dones[batch_indices], old_log_probs[batch_indices]
                )

    def update_step(self, states, actions, rewards, next_states, dones, old_log_probs):
        # Ensure tensors are detached from any previous computation graphs
        states, actions, rewards, next_states, dones, old_log_probs = (
            states.detach(), actions.detach(), rewards.detach(),
            next_states.detach(), dones.detach(), old_log_probs.detach()
        )

        # Calculate values and log_probs for the new policy
        values, log_probs, entropy = self.policy.evaluate(states, actions)
        values = values.squeeze()
        
        # Calculate the advantages
        advantages = rewards + self.gamma * (1 - dones) * values - values.detach()
        
        # Calculate ratios for the PPO objective
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # PPO loss
        loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(values, rewards) - 0.01 * entropy.mean()

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)  # retain_graph=True to prevent double-backward errors
        self.optimizer.step()

