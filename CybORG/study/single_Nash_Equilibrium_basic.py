import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Agent
class NashEquilibriumAgent:
    def __init__(self, observation_space, action_space_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, batch_size=64, memory_size=10000):
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        # Q-networks and optimizer
        self.q_network = QNetwork(observation_space.shape[0], action_space_size)
        self.target_network = QNetwork(observation_space.shape[0], action_space_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.memory[idx] for idx in batch))
        
        # Convert to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q-learning update using Double DQN logic
        q_values = self.q_network(states).gather(1, actions).squeeze()
        next_actions = torch.argmax(self.q_network(next_states), dim=1)
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute the loss and update the Q-network
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Define the environment rules and rewards
def compute_reward(state, action):
    """
    Example rules:
    - Reward +10 for action leading to Nash equilibrium.
    - Reward -10 for deviating from Nash equilibrium.
    """
    if action == 0:  # Example: Action 0 is the Nash equilibrium
        return 10
    else:
        return -10

# Main Training Function
def train_agent():
    # Create results folder
    results_dir = "single_Nash_Equilibrium_result"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize environment and agent
    observation_space = torch.zeros(10)  # Example state space
    action_space_size = 5  # Number of possible actions
    agent = NashEquilibriumAgent(observation_space, action_space_size)

    num_episodes = 500
    target_update_interval = 10
    reward_history = []
    cumulative_rewards = []

    # Training loop
    for episode in range(num_episodes):
        state = torch.rand(10)  # Randomly generated state for simplicity
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state.numpy())
            next_state = torch.rand(10)  # Example transition to next state
            reward = compute_reward(state, action)
            total_reward += reward
            done = np.random.rand() < 0.1  # Randomly terminate (10% chance)

            # Store transition and update policy
            agent.store_transition(state.numpy(), action, reward, next_state.numpy(), done)
            agent.update_policy()
            state = next_state

        # Update the target network
        if episode % target_update_interval == 0:
            agent.update_target_network()

        # Track and log rewards
        reward_history.append(total_reward)
        cumulative_rewards.append(sum(reward_history))
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Save results
    results = pd.DataFrame({
        "Episode": range(1, num_episodes + 1),
        "Reward": reward_history,
        "Cumulative Reward": cumulative_rewards
    })
    csv_path = os.path.join(results_dir, "training_rewards_basic.csv")
    results.to_csv(csv_path, index=False)

    # Plot and save reward graph
    # Ensure the plot doesn't block execution
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label="Total Reward", linewidth=2)
    plt.plot(cumulative_rewards, label="Cumulative Reward", linestyle="--", linewidth=2)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.title("Training Rewards (Basic)", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()  # Ensures no clipping of labels or title
    plt_path = os.path.join(results_dir, "training_rewards_basic.png")
    plt.savefig(plt_path, dpi=300, bbox_inches="tight", facecolor="white")  # Clean, high-resolution plot
    plt.close()  # Ensure the plot doesn't block execution
    # Save model
    model_path = os.path.join(results_dir, "nash_equilibrium_agent_model_basic.pth")
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"Results saved to: {csv_path} and {plt_path}")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train_agent()

