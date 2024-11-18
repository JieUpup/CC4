# worker_agent.py
import numpy as np
import torch
from parameter_server import ParameterServer

class WorkerAgent:
    def __init__(self, env, parameter_server, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.parameter_server = parameter_server
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space('blue_agent_0').n)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.parameter_server.q_network(state)).item()

    def interact_and_learn(self, num_episodes=100):
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            state = obs['blue_agent_0']
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                actions = {'blue_agent_0': action}
                obs, reward, terminated, truncated, _ = self.env.step(actions)
                next_state = obs['blue_agent_0']
                
                done = terminated['blue_agent_0'] or truncated['blue_agent_0']
                
                experience = (state, action, reward['blue_agent_0'], next_state, done)
                self.parameter_server.store_experience(experience)
                
                state = next_state
                total_reward += reward['blue_agent_0']
                
            # Decay epsilon after each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Sync worker Q-network with parameter server every few episodes
            if episode % 10 == 0:
                self.parameter_server.update_target_network()

