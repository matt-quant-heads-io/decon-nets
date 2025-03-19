"""
What is this script testing?
============================
This is testing the case where we are randomly adding difficulty*map_size state-action-pairs to a training set of
trajectories via randomly noising from an initial goal map. The number of episodes per difficulty is fixed at 10,000, 
and therefore, the number of state-action pairs difficult is difficulty*map_size*num_episodes.

This script trains a construction network using random sampling and plots the loss, 
entropy and correct % curves.

The training takes place across levels of difficulty.

Line 98 is where the random sampling takes place. For the Decon network experiments, we would replace the random sample
with instead using the decon network to predict high entropy states and use these states to add to the trajectory. 

Instead of the "done" check on line 121, we want a check on the average entropy over the trials being less than the "solved difficulty threshold".


"""

import os
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models import ConstructionNetwork, DestructionNetwork
from pcgrl_wrappers import PCGRLNoiseWrapper
from utils import calculate_entropy, evaluate_solvability, evaluate_diversity
import torch.nn.functional as F
from inference import infer
import time
import json

# Configuration parameters
CONFIG = {
    # PCGRL Environment
    'pcgrl_game': 'zelda',
    'pcgrl_representation': 'narrow',
    'initial_difficulty': 0.10,
    'difficulty_levels': [0.40, 0.80],
    
    # Networks
    'grid_size': (7, 11),  # Example size, adjust based on your PCGRL env
    'crop_size': (22, 22),  # Size of observation window
    'num_tile_types': 8,
    'learning_rate_construction': 1e-4,
    'learning_rate_destruction': 1e-4,
    
    # Training
    'num_state_action_pairs_per_difficulty': 200000,
    'evaluate_every_num_epochs': 20,
    'entropy_threshold': 0.03,
    'num_epochs_per_difficulty': 60,
    'batch_size': 32,
    'inference_trials': 30,
    'initial_dataset_size': 100,  # For initial destruction network training

    # Directories
    'warmup_dir': 'warmup',
    'random_agent_dir': 'random_agent_experiments',
    'common_baseline_dir': 'common_baseline', # NOTE: This contains the starting checkpoints for running the random and decon experiments
    'baseline_dir': 'baseline',
    'decon_dir': 'decon_experiments',
}

# Create required directories
os.makedirs(CONFIG['common_baseline_dir'], exist_ok=True)
os.makedirs(CONFIG['random_agent_dir'], exist_ok=True)
os.makedirs(CONFIG['decon_dir'], exist_ok=True)



# Create a wrapper class to make the construction network compatible with the inference interface
class ConstructionNetworkWrapper:
    def __init__(self, network):
        self.policy = self
        self.network = network
        
    def predict(self, state, x, y):
        state = F.one_hot(torch.tensor(transform(state, x, y, CONFIG['crop_size'])).long().unsqueeze(0), num_classes=8).float()
        state = state.cuda()
        
        # Get network prediction
        with torch.no_grad():
            output = self.network(state)
            
            # Calculate entropy from output probabilities
            probs = F.softmax(output, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            
            action = torch.argmax(output, dim=1)
            
        return action.cpu(), entropy.item()  # Return action and entropy



def transform(obs, x, y, obs_size):
    map = obs
    size = obs_size[0]
    pad = obs_size[0] // 2
    padded = np.pad(map, pad, constant_values=1)
    cropped = padded[y: y + size, x: x + size]
    
    return cropped


def generate_random_trajectory(env, difficulty, grid_size, trained_network):
    """
    Generate a trajectory from an environment object.
    
    Args:
        env: The environment object to generate the trajectory from
        difficulty (float): Difficulty factor that affects trajectory length
        grid_size (int): Base grid size for calculating trajectory length
        
    Returns:
        list: A list of (state, action, reward, next_state, done) tuples
    """
    trajectory = []
    trajectory_length = int(difficulty * grid_size[0] * grid_size[1])
    
    # Reset the environment to get initial state
    state_pos_dict, info = env.unwrapped.reset(target_map="./goal_maps/zelda/1.txt")
    y, x = state_pos_dict['pos'] 
    y, x = y.item(), x.item()
    state = state_pos_dict['map'] 
    
    for _ in range(trajectory_length):
        step_dict = {}
        step_dict['state'] = state
        step_dict['x'], step_dict['y'] = x, y   

        # Compute entropy using trained network if provided
        if trained_network is not None:
            # Convert state to tensor and get network prediction
            state_tensor = F.one_hot(torch.tensor(transform(state, x, y, CONFIG['crop_size'])).long(), num_classes=8).float().unsqueeze(0).cuda()
            with torch.no_grad():
                output = trained_network(state_tensor)
                probs = F.softmax(output, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
                step_dict['entropy'] = entropy.item()

        # Sample a random action from the environment's action space
        # NOTE: here is the random sample
        action = env.action_space.sample()
        step_dict['action'] = action.item()
        
        # Take the action in the environment
        next_state_pos_dict, reward, done, truncated, info = env.step(action)

        step_dict['reward'] = reward
        step_dict['done'] = done
        next_state = next_state_pos_dict['map']
        y, x = next_state_pos_dict['pos']
        x, y, = x.item(), y.item()        
        # Store the transition
        trajectory.append(step_dict)
        
        # Update state for next iteration
        state = next_state
            
    return trajectory


def train_destruction_network(env, difficulty, grid_size, num_state_action_pairs, trained_construction_network):
    """
    Train the destruction network using trajectories generated with construction network entropy.
    
    Args:
        env: The environment object to generate the trajectory from
        difficulty (float): Difficulty factor that affects trajectory length
        grid_size (int): Base grid size for calculating trajectory length
        num_state_action_pairs (int): Number of state-action pairs to collect
        trained_construction_network: Pre-trained construction network for generating entropy labels
        
    Returns:
        DestructionNetwork: The trained destruction network
    """
    # Create destruction network
    destruction_net = DestructionNetwork(grid_size, CONFIG['num_tile_types']).cuda()
    optimizer = torch.optim.Adam(destruction_net.parameters(), lr=CONFIG['learning_rate_destruction'])
    criterion = nn.MSELoss()

    # Generate trajectories using construction network entropy
    trajectories = []
    total_pairs = 0
    while total_pairs < num_state_action_pairs:
        trajectory = generate_random_trajectory(env, difficulty, grid_size, trained_construction_network)
        total_pairs += len(trajectory)
        trajectories.append(trajectory)

    # Create training dataset
    all_states = []
    all_entropies = []
    
    # Process each trajectory to get states and their stored entropies
    for traj in trajectories:
        for step in traj:
            # Get state and convert to tensor
            state_tensor = F.one_hot(torch.tensor(transform(step['state'], step['x'], step['y'], CONFIG['crop_size'])).long(), num_classes=8).float().unsqueeze(0).cuda()
            
            # Use the pre-computed entropy from the trajectory
            entropy = step['entropy']
            
            all_states.append(state_tensor)
            all_entropies.append(entropy)

    # Convert to tensors
    states = torch.cat(all_states)
    entropies = torch.tensor(all_entropies).float().cuda()

    # Training loop
    num_epochs = CONFIG['num_epochs_per_difficulty']
    batch_size = CONFIG['batch_size']
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Process in batches
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            batch_entropies = entropies[i:i + batch_size]

            # Forward pass
            predicted_entropies = destruction_net(batch_states)
            loss = criterion(predicted_entropies.squeeze(), batch_entropies)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Print progress
        avg_loss = epoch_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % CONFIG['evaluate_every_num_epochs'] == 0:
            torch.save(
                destruction_net.state_dict(),
                f"./{CONFIG['decon_dir']}/destruction_net_difficulty_{difficulty}_epoch_{epoch+1}.pth"
            )
            # Track destruction network loss
            destruction_losses.append(avg_loss)
    
    return destruction_net


def generate_decon_trajectory(env, difficulty, grid_size, trained_network):
    """
    Generate a trajectory from an environment object using a trained neural network
    that selects actions based on predicted state entropy.
    
    Args:
        env: The environment object to generate the trajectory from
        difficulty (float): Difficulty factor that affects trajectory length
        grid_size (int): Base grid size for calculating trajectory length
        trained_network: A pre-trained construction network for entropy prediction
        
    Returns:
        list: A list of state, action, reward, next_state, done dictionaries
    """
    trajectory = []
    trajectory_length = int(difficulty * grid_size[0] * grid_size[1])
    
    # Reset the environment to get initial state
    state_pos_dict, info = env.unwrapped.reset(target_map="./goal_maps/zelda/1.txt")
    y, x = state_pos_dict['pos'] 
    y, x = y.item(), x.item()
    state = state_pos_dict['map'] 
    
    for _ in range(trajectory_length):
        step_dict = {}
        step_dict['state'] = state
        step_dict['x'], step_dict['y'] = x, y   

        # Use the trained network to select the action with highest predicted entropy
        action_entropies = []
        possible_actions = range(env.action_space.n)  # Assuming Discrete action space
        
        # For each possible action, predict the entropy of the resulting state
        for a in possible_actions:
            # Create a copy of the environment to simulate the action
            env_copy = env.unwrapped.clone()
            next_state_pos_dict, _, _, _, _ = env_copy.step(a)
            next_state = next_state_pos_dict['map']
            
            # Convert state to network input format
            state_tensor = F.one_hot(torch.tensor(transform(next_state, x, y, CONFIG['crop_size'])).long(), num_classes=8).float().unsqueeze(0).cuda()
            
            # Get network prediction
            with torch.no_grad():
                output = trained_network(state_tensor)
                
                # Calculate entropy from output probabilities
                probs = F.softmax(output, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
                
            action_entropies.append(entropy.item())
        
        # Select the action with the highest predicted entropy
        action = possible_actions[action_entropies.index(max(action_entropies))]
        step_dict['action'] = action
        
        # Take the selected action in the environment
        next_state_pos_dict, reward, done, truncated, info = env.step(action)

        step_dict['reward'] = reward
        step_dict['done'] = done
        next_state = next_state_pos_dict['map']
        y, x = next_state_pos_dict['pos']
        x, y, = x.item(), y.item()        
        # Store the transition
        trajectory.append(step_dict)
        
        # Update state for next iteration
        state = next_state
            
    return trajectory



# NOTE: GLOBAL VARS FOR TRAJECTORY GENERATION, TRAINING, EVALUATION
# Create initial PCGRL environment with 5% difficulty
env = gym.make(f"{CONFIG['pcgrl_game']}-{CONFIG['pcgrl_representation']}-v0")
env = PCGRLNoiseWrapper(env, difficulty_pct=CONFIG['initial_difficulty'])

# NOTE: TRAINING AND EVALUATION (EVAL IS INSIDE THE TRAINING LOOP)
# Create and train the construction network
construction_net = ConstructionNetwork(CONFIG['grid_size'], CONFIG['num_tile_types']).cuda()

# Load the latest checkpoint for the construction network
latest_epoch = CONFIG['num_epochs_per_difficulty']
construction_net.load_state_dict(torch.load(f"./{CONFIG['common_baseline_dir']}/construction_net_difficulty_{CONFIG['difficulty_levels'][-1]}_epoch_{latest_epoch}.pth"))
construction_net.eval()  # Set to evaluation mode

destruction_net = DestructionNetwork(CONFIG['grid_size'], CONFIG['num_tile_types']).cuda()
optimizer = torch.optim.Adam(destruction_net.parameters(), lr=CONFIG['learning_rate_construction'])
criterion = nn.MSELoss()

# Dictionary to store metrics for each difficulty
metrics_per_difficulty = {
    'destruction_losses': {},
    'inference_entropies': {},
    'epoch_times': {},  # Add timing metrics
}

# NOTE: TRAJECTORY GENERATION AND TRAINING SECTION
for difficulty in CONFIG['difficulty_levels']:
    print(f"\nTraining at difficulty level {difficulty}")
    
    # Train the destruction network using the loaded construction network
    print("Training Destruction Network...")
    destruction_net = train_destruction_network(env, difficulty, CONFIG['grid_size'], CONFIG['num_state_action_pairs_per_difficulty'], construction_net)
    
    # Lists to store metrics for current difficulty
    destruction_losses = []
    mean_entropies = []
    epoch_times = []

    # Store metrics for this difficulty level
    metrics_per_difficulty['destruction_losses'][difficulty] = destruction_losses
    metrics_per_difficulty['inference_entropies'][difficulty] = mean_entropies
    metrics_per_difficulty['epoch_times'][difficulty] = epoch_times

# Plot all metrics with curves for each difficulty
plt.figure(figsize=(15, 12))

# Plot Destruction Network Loss
plt.subplot(2, 2, 1)
for diff, loss_values in metrics_per_difficulty['destruction_losses'].items():
    plt.plot(loss_values, label=f'Difficulty {diff}')
plt.title('Destruction Network Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Inference Entropies
plt.subplot(2, 2, 2)
for diff, entropy_values in metrics_per_difficulty['inference_entropies'].items():
    plt.plot(entropy_values, label=f'Difficulty {diff}')
plt.title('Inference Entropy')
plt.xlabel('Epoch')
plt.ylabel('Entropy')
plt.legend()

# Plot Epoch Times
plt.subplot(2, 2, 3)
for diff, time_values in metrics_per_difficulty['epoch_times'].items():
    plt.plot(time_values, label=f'Difficulty {diff}')
plt.title('Epoch Time')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['decon_dir'], f'destruction_training_metrics.png'))
plt.close()

# Save metrics data to JSON file
metrics_json = {
    "metrics_per_difficulty": {
        "destruction_losses": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['destruction_losses'].items()},
        "inference_entropies": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['inference_entropies'].items()},
        "epoch_times": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['epoch_times'].items()}
    },
    "config": CONFIG
}

# Save to JSON file
with open(os.path.join(CONFIG['decon_dir'], 'destruction_training_metrics.json'), 'w') as f:
    json.dump(metrics_json, f, indent=4)


