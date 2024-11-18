# distributed_double_dqn.py
import multiprocessing
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from parameter_server import ParameterServer
from worker_agent import WorkerAgent

def worker_process(worker_id, param_server):
    sg = EnterpriseScenarioGenerator()
    cyborg = CybORG(scenario_generator=sg)
    env = BlueFlatWrapper(env=cyborg)
    
    agent = WorkerAgent(env, param_server)
    agent.interact_and_learn(num_episodes=100)

def main():
    # Initialize environment and parameter server
    sg = EnterpriseScenarioGenerator()
    cyborg = CybORG(scenario_generator=sg)
    env = BlueFlatWrapper(env=cyborg)

    param_server = ParameterServer(observation_space=env.observation_space('blue_agent_0'),
                                   action_space=env.action_space('blue_agent_0'))
    
    # Number of parallel workers
    num_workers = 4

    # Create and start worker processes
    processes = []
    for worker_id in range(num_workers):
        p = multiprocessing.Process(target=worker_process, args=(worker_id, param_server))
        processes.append(p)
        p.start()

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Save the trained model
    torch.save(param_server.q_network.state_dict(), 'distributed_double_dqn_model.pth')

if __name__ == "__main__":
    main()

