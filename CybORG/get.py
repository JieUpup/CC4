
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFixedActionWrapper
from rich import print

# Step 1: Create an EnterpriseScenarioGenerator instance
sg = EnterpriseScenarioGenerator()

# Step 2: Instantiate the CybORG environment with the scenario generator
cyborg = CybORG(scenario_generator=sg)

# Step 3: Wrap the environment for Blue agents using BlueFixedActionWrapper
env = BlueFixedActionWrapper(env=cyborg)

# Step 4: Reset the environment and get the initial observation
obs, _ = env.reset()

# Step 5: Display available Blue agents
print("[bold cyan]Available Blue Agents:[/bold cyan]")
print(list(obs.keys()))  # Output: dict_keys(['blue_agent_0', 'blue_agent_1', ..., 'blue_agent_4'])

# Step 6: Iterate through each Blue agent's observation
for agent, agent_obs in obs.items():
    print(f"\n[bold green]Observations for {agent}:[/bold green]")

    # Step 7: Display host-level observations
    for host, details in agent_obs.items():
        print(f"\n[bold yellow]Host: {host}[/bold yellow]")

        # Ensure 'details' is a dictionary before accessing its keys
        if isinstance(details, dict):
            # Print interface information
            interfaces = details.get('Interface', [])
            if interfaces:
                print("[bold magenta]Interfaces:[/bold magenta]")
                for interface in interfaces:
                    print(f"  - Interface: {interface['interface_name']}, "
                          f"IP: {interface['ip_address']}, Subnet: {interface['Subnet']}")
            
            # Print session details
            sessions = details.get('Sessions', [])
            if sessions:
                print("[bold magenta]Sessions:[/bold magenta]")
                for session in sessions:
                    print(f"  - Session ID: {session['session_id']}, "
                          f"User: {session['username']}, PID: {session['PID']}, Agent: {session['agent']}")

            # Print process information
            processes = details.get('Processes', [])
            if processes:
                print("[bold magenta]Processes:[/bold magenta]")
                for process in processes:
                    print(f"  - Process ID: {process['PID']}, User: {process['username']}")
            
            # Print user info
            user_info = details.get('User Info', [])
            if user_info:
                print("[bold magenta]User Info:[/bold magenta]")
                for user in user_info:
                    username = user.get('username', 'Unknown')
                    groups = user.get('Groups', [])
                    group_ids = [group['GID'] for group in groups]
                    print(f"  - User: {username}, Groups: {group_ids}")
            
            # Print system info
            system_info = details.get('System info', {})
            if system_info:
                print("[bold magenta]System Info:[/bold magenta]")
                print(f"  - Hostname: {system_info.get('Hostname')}, "
                      f"OS: {system_info.get('OSType')}, "
                      f"Distribution: {system_info.get('OSDistribution')}, "
                      f"Architecture: {system_info.get('Architecture')}, "
                      f"Position: {system_info.get('position')}")
        else:
            # Handle non-dictionary 'details' (e.g., TernaryEnum)
            print(f"[red]Unexpected data type: {type(details)}[/red], Value: {details}")

# Step 8: Example: Accessing specific information from Blue agent 0
blue_agent_0_obs = obs.get('blue_agent_0', {})
router_obs = blue_agent_0_obs.get('restricted_zone_a_subnet_router', {})
if isinstance(router_obs, dict):
    print("\n[bold magenta]Detailed observation for blue_agent_0 - restricted_zone_a_subnet_router:[/bold magenta]")
    print(router_obs)
else:
    print(f"[red]Unexpected data type for 'restricted_zone_a_subnet_router': {type(router_obs)}[/red], Value: {router_obs}")

# Step 9: Display action space and action labels for 'blue_agent_0'
print("\n[bold cyan]Action Space for blue_agent_0:[/bold cyan]")
print(env.action_space('blue_agent_0'))

print("\n[bold cyan]Action Labels for blue_agent_0:[/bold cyan]")
print(env.action_labels('blue_agent_0'))

