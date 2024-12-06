import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from rl_env_setup import setup_rl_env  # Import your environment setup function
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Initialize the environment and wrap it for RL
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
rollout_buffer = deque(maxlen=10000)  # Fixed-size buffer for rollouts
reward_scaling = 1.0  # Scaling factor for rewards
log_dir = "ppo_training_logs"
writer = SummaryWriter(log_dir)

# Storage for results
results = {"Episode": [], "Reward": [], "Steps": [], "Entropy": []}

# Training loop
for episode in range(1, num_episodes + 1):
    obs, _ = env.reset()
    state = obs['blue_agent_0']
    done = False
    episode_reward = 0
    step_count = 0

    while not done:
        # Select an action using the agent's policy
        action, log_prob = agent.select_action(state)

        # Perform the action in the environment
        actions = {'blue_agent_0': action}
        obs, reward, terminated, truncated, info = env.step(actions)
        next_state = obs['blue_agent_0']
        reward = reward['blue_agent_0'] / reward_scaling  # Normalize reward
        done = terminated['blue_agent_0'] or truncated['blue_agent_0']

        # Store the transition in the rollout buffer
        rollout_buffer.append((state, action, reward, next_state, done, log_prob))

        # Update the state and track episode reward
        state = next_state
        episode_reward += reward
        step_count += 1

    # Collect rollout data from buffer
    batch_size = min(len(rollout_buffer), batch_size)
    sampled_data = [rollout_buffer[idx] for idx in np.random.choice(len(rollout_buffer), batch_size, replace=False)]
    states, actions, rewards, next_states, dones, old_log_probs = zip(*sampled_data)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(np.array(dones), dtype=torch.float32)
    old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32)

    # Perform PPO policy update
    agent.update_policy((states, actions, rewards, next_states, dones, old_log_probs), batch_size, update_epochs)

    # Log episode performance
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = agent.policy.actor(state_tensor)  # Get action probabilities
        dist = torch.distributions.Categorical(action_probs)  # Create categorical distribution
        policy_entropy = dist.entropy().mean().item()  # Compute entropy

    # Save results to log
    results["Episode"].append(episode)
    results["Reward"].append(episode_reward)
    results["Steps"].append(step_count)
    results["Entropy"].append(policy_entropy)

    writer.add_scalar("Reward/Episode", episode_reward, episode)
    writer.add_scalar("Steps/Episode", step_count, episode)
    writer.add_scalar("Policy/Entropy", policy_entropy, episode)  # Log entropy
    print(f"Episode {episode}/{num_episodes}, Total Reward: {episode_reward:.2f}, Steps: {step_count}, Entropy: {policy_entropy:.2f}")

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_csv_path = "training_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Training results saved to {results_csv_path}")

# Plot the results and save as PNG
plt.figure(figsize=(12, 6))
plt.plot(results["Episode"], results["Reward"], label="Reward")
plt.plot(results["Episode"], results["Steps"], label="Steps")
plt.xlabel("Episode")
plt.ylabel("Values")
plt.title("Training Performance")
plt.legend()
results_png_path = "training_results.png"
plt.savefig(results_png_path)
plt.close()
print(f"Training plot saved to {results_png_path}")

# Save the agent's policy
model_path = "ppo_agent_policy.pth"
torch.save(agent.policy.state_dict(), model_path)  # Save the PPO policy model
print(f"Model saved to {model_path}")

# Close the writer
writer.close()

