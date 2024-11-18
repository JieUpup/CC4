# train_ppo.py

import numpy as np
import torch
from ppo_agent import PPOAgent
from environment_setup import setup_rl_env  # Import your environment setup function

# Initialize CybORG environment and wrap it for RL
env, _ = setup_rl_env()

# Initialize PPOAgent
agent = PPOAgent(
    observation_space=env.observation_space('blue_agent_0'), 
    action_space=env.action_space('blue_agent_0')
)

# Training parameters
num_episodes = 500
batch_size = 64
update_epochs = 10
rollout_data = []

# Training loop
for episode in range(num_episodes):
    obs, _ = env.reset()
    state = obs['blue_agent_0']
    done = False
    episode_reward = 0

    while not done:
        # Select an action using the agent's policy
        action, log_prob = agent.select_action(state)
        
        # Perform the action and get the next observation, reward, and done signal
        actions = {'blue_agent_0': action}
        obs, reward, terminated, truncated, info = env.step(actions)
        next_state = obs['blue_agent_0']
        
        # Store the transition for PPO update
        rollout_data.append((state, action, reward['blue_agent_0'], next_state, terminated['blue_agent_0'], log_prob))
        state = next_state
        episode_reward += reward['blue_agent_0']
        
        # Check if the episode is done
        done = terminated['blue_agent_0'] or truncated['blue_agent_0']
    
    # Organize the rollout data for the current episode
    states, actions, rewards, next_states, dones, old_log_probs = zip(*rollout_data)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(np.array(dones), dtype=torch.float32)
    old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32)

    # Clear the rollout data after preparing
    rollout_data = []

    # Perform PPO update
    agent.update_policy((states, actions, rewards, next_states, dones, old_log_probs), batch_size, update_epochs)
    
    # Print the episode result
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward:.2f}")

