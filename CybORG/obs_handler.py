# obs_handler.py
from rich import print

def display_obs(obs):
    print("[bold cyan]Available Blue Agents:[/bold cyan]")
    print(list(obs.keys()))  # Display available Blue agents

    # Iterate through each Blue agent's observation
    for agent, agent_obs in obs.items():
        print(f"\n[bold green]Observations for {agent}:[/bold green]")

        # Display host-level observations
        for host, details in agent_obs.items():
            print(f"\n[bold yellow]Host: {host}[/bold yellow]")

            if isinstance(details, dict):
                # Display interface information
                interfaces = details.get('Interface', [])
                if interfaces:
                    print("[bold magenta]Interfaces:[/bold magenta]")
                    for interface in interfaces:
                        print(f"  - Interface: {interface['interface_name']}, "
                              f"IP: {interface['ip_address']}, Subnet: {interface['Subnet']}")

                # Display session details
                sessions = details.get('Sessions', [])
                if sessions:
                    print("[bold magenta]Sessions:[/bold magenta]")
                    for session in sessions:
                        print(f"  - Session ID: {session['session_id']}, User: {session['username']}, "
                              f"PID: {session['PID']}, Agent: {session['agent']}")

                # Display process information
                processes = details.get('Processes', [])
                if processes:
                    print("[bold magenta]Processes:[/bold magenta]")
                    for process in processes:
                        print(f"  - Process ID: {process['PID']}, User: {process['username']}")

                # Display user info
                user_info = details.get('User Info', [])
                if user_info:
                    print("[bold magenta]User Info:[/bold magenta]")
                    for user in user_info:
                        username = user.get('username', 'Unknown')
                        groups = user.get('Groups', [])
                        group_ids = [group['GID'] for group in groups]
                        print(f"  - User: {username}, Groups: {group_ids}")

                # Display system info
                system_info = details.get('System info', {})
                if system_info:
                    print("[bold magenta]System Info:[/bold magenta]")
                    print(f"  - Hostname: {system_info.get('Hostname')}, OS: {system_info.get('OSType')}, "
                          f"Distribution: {system_info.get('OSDistribution')}, "
                          f"Architecture: {system_info.get('Architecture')}, "
                          f"Position: {system_info.get('position')}")
            else:
                # Handle non-dictionary 'details'
                print(f"[red]Unexpected data type: {type(details)}[/red], Value: {details}")

