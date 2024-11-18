# take_action.py
import numpy as np
from rich import print

def take_action(env, action, message=None):
    # Execute the action and step the environment
    if message is None:
        obs, reward, terminated, truncated, info = env.step(action)
    else:
        obs, reward, terminated, truncated, info = env.step(action, messages=message)

    return obs, reward, terminated, truncated, info

def display_action_results(action, reward, obs, agent='blue_agent_0'):
    print(f"\n[bold cyan]Action taken by {agent}:[/bold cyan] {action[agent]}")
    print(f"[bold cyan]Reward for {agent}:[/bold cyan] {reward[agent]}")

    if 'message' in obs[agent]:
        print(f"[bold cyan]Message received by {agent}:[/bold cyan] {obs[agent]['message']}")

