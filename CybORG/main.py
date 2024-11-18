# main.py
from env_setup import initialize_env
from obs_handler import display_obs
from take_action import take_action, display_action_results

def main():
    # Initialize environment
    env, obs = initialize_env()

    # Display initial observations
    display_obs(obs)

    # Example action: 'Restore restricted_zone_a_subnet_user_host_3'
    action = {'blue_agent_0': 42}
    obs, reward, terminated, truncated, info = take_action(env, action)

    # Display action results
    display_action_results(action, reward, obs)

    # Example message passing
    import numpy as np
    message = {'blue_agent_0': np.array([1, 0, 0, 0, 0, 0, 0, 0])}
    obs, reward, terminated, truncated, info = take_action(env, action, message)

    # Display updated observations after message passing
    display_action_results(action, reward, obs)

if __name__ == "__main__":
    main()

