# env_setup.py
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFixedActionWrapper

def initialize_env():
    # Step 1: Create an EnterpriseScenarioGenerator instance
    sg = EnterpriseScenarioGenerator()

    # Step 2: Instantiate the CybORG environment with the scenario generator
    cyborg = CybORG(scenario_generator=sg)

    # Step 3: Wrap the environment for Blue agents
    env = BlueFixedActionWrapper(env=cyborg)

    # Step 4: Reset the environment and get the initial observation
    obs, _ = env.reset()

    return env, obs

