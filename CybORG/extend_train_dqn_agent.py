import numpy as np
import torch
from dqn_agent import DQNAgent
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper

# Initialize CybORG environment and wrap it for RL
sg = EnterpriseScenarioGenerator()
cyborg = CybORG(scenario_generator=sg)
env = BlueFlatWrapper(env=cyborg)

# Assume agent names are like blue_agent_0, blue_agent_1, etc.
agent_names = env.get_agent_names()
agents = {}

# Initialize multiple DQNAgents
for name in agent_names:
    agents[name] = DQNAgent(observation_space=env.observation_space(name), action_space=env.action_space(name))

# Training parameters
num_episodes = 500
target_update_interval = 10  # Update target network every 10 episodes

# Training loop
for episode in range(num_episodes):
    obs, _ = env.reset()
    total_rewards = {name: 0 for name in agent_names}
    dones = {name: False for name in agent_names}

    while not any(dones.values()):
        actions = {}

        for name in agent_names:
            if not dones[name]:
                # Select an action using the agent's policy
                state = obs[name]
                action = agents[name].select_action(state)
                actions[name] = action

        # Perform the actions and get the next observations, rewards, and done signals
        obs, rewards, terminated, truncated, info = env.step(actions)

        for name in agent_names:
            if not dones[name]:
                next_state = obs[name]
                reward = rewards[name]
                done = terminated[name] or truncated[name]

                # Store the transition in the replay memory
                agents[name].store_transition(state, actions[name], reward, next_state, done)

                # Update the policy using the sampled transitions
                agents[name].update_policy()

                # Update rewards and done status
                total_rewards[name] += reward
                dones[name] = done

    # Update the target network for each agent every few episodes
    if episode % target_update_interval == 0:
        for agent in agents.values():
            agent.update_target_network()

    # Print the episode result for each agent
    for name, total_reward in total_rewards.items():
        print(f"Agent {name}, Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agents[name].epsilon:.2f}")

