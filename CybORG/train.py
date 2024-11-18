# train.py
import numpy as np
from rl_agent import BasicRLAgent

def train_agent(env, num_episodes=100):
    # Initialize the RL agent
    agent = BasicRLAgent(env.action_space('blue_agent_0'))

    for episode in range(num_episodes):
        # Reset environment and get initial observation
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Agent selects an action
            action = agent.select_action(obs['blue_agent_0'])

            # Take the action in the environment
            actions = {'blue_agent_0': action}
            next_obs, reward, terminated, truncated, info = env.step(actions)

            # Update the agent's policy
            agent.update_policy(obs['blue_agent_0'], action, reward['blue_agent_0'], 
                                next_obs['blue_agent_0'], terminated['blue_agent_0'])

            # Update observation and cumulative reward
            obs = next_obs
            total_reward += reward['blue_agent_0']

            # Check if the episode has ended
            done = terminated['blue_agent_0'] or truncated['blue_agent_0']

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print("Training completed.")

