import torch
from single_Nash_Equilibrium import QNetwork 

model_path = "single_Nash_Equilibrium_result/nash_equilibrium_agent_model.pth"
observation_space_dim = 10  # Replace with your actual input dimension
action_space_size = 5       # Replace with your actual number of actions

model = QNetwork(observation_space_dim, action_space_size)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode
print("Model loaded successfully.")

