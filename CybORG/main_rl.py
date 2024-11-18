# main_rl.py
from rl_env_setup import setup_rl_env
from train import train_agent

def main():
    # Setup RL environment
    env, obs = setup_rl_env()

    # Print initial observation space and observation
    from rich import print
    print(f"Space: {env.observation_space('blue_agent_0')}\n")
    print(f"Observation: {obs['blue_agent_0']}\n")

    # Train the RL agent
    train_agent(env, num_episodes=100)

if __name__ == "__main__":
    main()

