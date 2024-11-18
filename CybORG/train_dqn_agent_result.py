import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from dqn_agent import DQNAgent
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper

# Set up result directory
result_dir = 'dqn_result'
os.makedirs(result_dir, exist_ok=True)

# Initialize CybORG environment and wrap it for RL
sg = EnterpriseScenarioGenerator()
cyborg = CybORG(scenario_generator=sg)
env = BlueFlatWrapper(env=cyborg)

# Initialize DQNAgent
agent = DQNAgent(observation_space=env.observation_space('blue_agent_0'), action_space=env.action_space('blue_agent_0'))

# Training parameters
num_episodes = 500
target_update_interval = 10  # Update target network every 10 episodes
rewards_per_episode = []

# Training loop
for episode in range(num_episodes):
    obs, _ = env.reset()
    state = obs['blue_agent_0']  # Get the initial observation for blue_agent_0
    total_reward = 0
    done = False

    while not done:
        # Select an action using the agent's policy
        action = agent.select_action(state)
        
        # Perform the action and get the next observation, reward, and done signal
        actions = {'blue_agent_0': action}
        obs, reward, terminated, truncated, info = env.step(actions)
        next_state = obs['blue_agent_0']
        
        # Store the transition in the replay memory
        agent.store_transition(state, action, reward['blue_agent_0'], next_state, terminated['blue_agent_0'])
        
        # Update the policy using the sampled transitions
        agent.update_policy()
        
        # Move to the next state
        state = next_state
        total_reward += reward['blue_agent_0']
        
        # Check if the episode is done
        done = terminated['blue_agent_0'] or truncated['blue_agent_0']

    # Store episode reward
    rewards_per_episode.append(total_reward)

    # Update the target network every few episodes
    if episode % target_update_interval == 0:
        agent.update_target_network()
    
    # Print the episode result
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# Save rewards to CSV
rewards_df = pd.DataFrame(rewards_per_episode, columns=['Reward'])
rewards_df.to_csv(os.path.join(result_dir, 'training_rewards.csv'), index_label='Episode')

# Plot the rewards
plt.figure(figsize=(10, 6))
plt.plot(rewards_per_episode, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Rewards')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'training_rewards.png'))
plt.show()

