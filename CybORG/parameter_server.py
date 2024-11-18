# parameter_server.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dqn_agent import QNetwork

class ParameterServer:
    def __init__(self, observation_space, action_space, lr=1e-3, gamma=0.99, batch_size=64, memory_size=100000):
        # Initialize Q-network and target network
        self.q_network = QNetwork(observation_space.shape[0], action_space.n)
        self.target_network = QNetwork(observation_space.shape[0], action_space.n)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Initialize global experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Sync target network initially
        self.update_target_network()

    def update_target_network(self):
        # Copy weights from Q-network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_experience(self, experience):
        # Add experience to replay buffer
        self.memory.append(experience)

    def sample_experiences(self):
        # Sample batch of experiences for training
        if len(self.memory) < self.batch_size:
            return None
        return random.sample(self.memory, self.batch_size)

    def update_q_network(self):
        # Train the Q-network with experiences from the replay buffer
        batch = self.sample_experiences()
        if not batch:
            return
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Double DQN update
        current_q_values = self.q_network(states).gather(1, actions).squeeze()
        next_actions = torch.argmax(self.q_network(next_states), dim=1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss and backpropagate
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

