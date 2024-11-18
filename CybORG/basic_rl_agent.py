# basic_rl_agent.py
import numpy as np

class BasicRLAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        # Random action selection (placeholder for real RL algorithms)
        return np.random.choice(self.action_space.n)

    def update_policy(self, obs, action, reward, next_obs, done):
        # Placeholder: Add your RL update logic (e.g., Q-learning or PPO update)
        pass

