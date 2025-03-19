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
import copy


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
    'evaluate_every_num_epochs': 100,
    'entropy_threshold': 0.03,
    'num_epochs_per_difficulty': 300,
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


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, crop_size, num_classes=8, device=None):
        self.states = []
        self.actions = []
        
        for traj in trajectories:
            for step in traj:
                state = F.one_hot(torch.tensor(transform(step['state'], step['x'], step['y'], crop_size)).long(), num_classes=num_classes)
                action = F.one_hot(torch.tensor([step['action']]), num_classes=num_classes)
                self.states.append(state)
                self.actions.append(action)
        
        self.states = torch.stack(self.states).float().to(device)
        self.actions = torch.stack(self.actions).float().to(device)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


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


def generate_random_trajectory(env, difficulty, grid_size):
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



# NOTE: GLOBAL VARS FOR TRAJECTORY GENERATION, TRAINING, EVALUATION
# Create initial PCGRL environment with 5% difficulty
env = gym.make(f"{CONFIG['pcgrl_game']}-{CONFIG['pcgrl_representation']}-v0")
env = PCGRLNoiseWrapper(env, difficulty_pct=CONFIG['initial_difficulty'])

# Dictionary to store metrics for each difficulty
metrics_per_difficulty = {
    'losses': {},
    'entropies': {},
    'action_correct_pcts': {},
    'solve_percentages': {},
    'diversity_scores': {},
    'inference_entropies': {},
    'epoch_times': {}  # Add timing metrics
}

# NOTE: TRAINING AND EVALUATION (EVAL IS INSIDE THE TRAINING LOOP)
# Create and train the construction network
torch.backends.cudnn.benchmark = True
construction_net = ConstructionNetwork(CONFIG['grid_size'], CONFIG['num_tile_types']).cuda()
optimizer = torch.optim.Adam(construction_net.parameters(), lr=CONFIG['learning_rate_construction'])
criterion = nn.CrossEntropyLoss()
construction_net.train()

for i, (name, param) in enumerate(construction_net.named_parameters()):
    if param in optimizer.state:
        state = optimizer.state[param]
        if 'exp_avg' in state:
            print(f"Parameter {name} exp_avg mean: {state['exp_avg'].mean().item()}")


dataset = None
dataloader = None

# NOTE: TRAJECTORY GENERATION SECTION
for difficulty in CONFIG['difficulty_levels']:
    trajectories = []
    num_state_action_pairs = 0
    while num_state_action_pairs <= CONFIG['num_state_action_pairs_per_difficulty']:
        trajectory = generate_random_trajectory(env, difficulty, CONFIG['grid_size'])
        num_state_action_pairs += len(trajectory)
        trajectories.append(trajectory)

    # Lists to store metrics for current difficulty
    losses = []
    entropies = []
    action_correct_pcts = []
    solve_percentages = []  # Add list for solve percentages
    diversity_scores = []   # Add list for diversity scores
    mean_entropies = []
    epoch_times = []  # Add list for epoch times

    # Create dataset and dataloader from trajectories
    dataset = TrajectoryDataset(trajectories, CONFIG['crop_size'], CONFIG['num_tile_types'], device='cuda')
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Training loop
    for epoch in range(CONFIG['num_epochs_per_difficulty']):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_entropy = 0
        num_batches = 0
        
        num_correct_per_epoch = 0
        batches_per_epoch = 0
        
        # Process in batches using dataloader
        for batch_states, batch_actions in dataloader:
            # batch_states = batch_states.cuda()
            # batch_actions = batch_actions.cuda()

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            # Forward pass
            outputs = construction_net(batch_states)
            
            # Calculate loss - treat next_states as ground truth
            loss = criterion(outputs, torch.squeeze(batch_actions, 1))
            
            # Calculate entropy of output distribution
            probs = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            num_correct_per_epoch += (torch.max(outputs, dim=1).indices == torch.max(torch.squeeze(batch_actions, 1), dim=1).indices).type(torch.float).sum().item()
            batches_per_epoch += len(outputs)
            
            
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_entropy += entropy.item()
            num_batches += 1
        
        # Average metrics for the epoch
        avg_loss = epoch_loss / num_batches
        avg_entropy = epoch_entropy / num_batches
        avg_correct_pct = num_correct_per_epoch / batches_per_epoch
        losses.append(avg_loss)
        entropies.append(avg_entropy)
        action_correct_pcts.append(avg_correct_pct)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f'Epoch [{epoch+1}/{CONFIG["num_epochs_per_difficulty"]}], Loss: {avg_loss:.4f}, Entropy: {avg_entropy:.4f}, Correct {avg_correct_pct:.4f}, Time: {epoch_time:.2f}s')
        
        # Every 100 epochs, evaluate the network's performance
        if (epoch + 1) % CONFIG['evaluate_every_num_epochs'] == 0:
            construction_net_copy = copy.deepcopy(construction_net)
            optimizer_copy = copy.deepcopy(optimizer)
            wrapped_network = ConstructionNetworkWrapper(construction_net_copy)
            solve_pct, div_score, mean_entropy = infer(
                wrapped_network,
                env,
                trials=CONFIG['inference_trials'],  # Reduced number of trials for faster evaluation during training
                verbose=False,
                div_cutoff=0.10,
                goal_map_path="./goal_maps/zelda/1.txt"
            )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': construction_net_copy.state_dict(),
                'optimizer_state_dict': optimizer_copy.state_dict(),
                'loss': loss,
                'difficulty': difficulty
            }

            torch.save(
                checkpoint,
                f"./{CONFIG['common_baseline_dir']}/difficulty_{difficulty}_epoch_{epoch+1}.pth",
            )
            solve_percentages.append(solve_pct)
            diversity_scores.append(div_score)
            mean_entropies.append(mean_entropy)
            print(f'Evaluation - Solve %: {solve_pct:.2f}, Diversity: {div_score:.4f}, Entropy: {mean_entropy:.4f}')

            for i, (name, param) in enumerate(construction_net.named_parameters()):
                if param in optimizer.state:
                    state = optimizer.state[param]
                    if 'exp_avg' in state:
                        print(f"Parameter {name} exp_avg mean: {state['exp_avg'].mean().item()}")

        
    # Store metrics for this difficulty level
    metrics_per_difficulty['losses'][difficulty] = losses
    metrics_per_difficulty['entropies'][difficulty] = entropies
    metrics_per_difficulty['action_correct_pcts'][difficulty] = action_correct_pcts
    metrics_per_difficulty['solve_percentages'][difficulty] = solve_percentages
    metrics_per_difficulty['diversity_scores'][difficulty] = diversity_scores
    metrics_per_difficulty['inference_entropies'][difficulty] = mean_entropies
    metrics_per_difficulty['epoch_times'][difficulty] = epoch_times


# Plot all metrics with curves for each difficulty
plt.figure(figsize=(15, 10))  # Make figure slightly taller to accommodate new plot

# Plot Training Loss
plt.subplot(2, 3, 1)
for diff, loss_values in metrics_per_difficulty['losses'].items():
    plt.plot(loss_values, label=f'Difficulty {diff}')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Output Entropy
plt.subplot(2, 3, 2)
for diff, entropy_values in metrics_per_difficulty['entropies'].items():
    plt.plot(entropy_values, label=f'Difficulty {diff}')
plt.title('Output Entropy')
plt.xlabel('Epoch')
plt.ylabel('Entropy')
plt.legend()

# Plot Correct Predictions
plt.subplot(2, 3, 3)
for diff, correct_values in metrics_per_difficulty['action_correct_pcts'].items():
    plt.plot(correct_values, label=f'Difficulty {diff}')
plt.title('Correct Predictions')
plt.xlabel('Epoch')
plt.ylabel('% Correct')
plt.legend()

# Plot Solve Percentages
plt.subplot(2, 3, 4)
epochs_evaluated = np.arange(0, CONFIG['num_epochs_per_difficulty'], CONFIG['evaluate_every_num_epochs'])
for diff, solve_values in metrics_per_difficulty['solve_percentages'].items():
    plt.plot(epochs_evaluated, solve_values, label=f'Difficulty {diff}')
plt.title('Solve Percentage')
plt.xlabel('Epoch')
plt.ylabel('Solve %')
plt.legend()

# Plot Diversity Scores
plt.subplot(2, 3, 5)
for diff, diversity_values in metrics_per_difficulty['diversity_scores'].items():
    plt.plot(epochs_evaluated, diversity_values, label=f'Difficulty {diff}')
plt.title('Diversity Score')
plt.xlabel('Epoch')
plt.ylabel('Diversity')
plt.legend()

# Plot Epoch Times
plt.subplot(2, 3, 6)
for diff, time_values in metrics_per_difficulty['epoch_times'].items():
    plt.plot(time_values, label=f'Difficulty {diff}')
plt.title('Epoch Time')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['common_baseline_dir'], f'results_per_difficult.png'))
plt.close()

# Save metrics data to JSON file
metrics_json = {
    "metrics_per_difficulty": {
        "losses": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['losses'].items()},
        "entropies": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['entropies'].items()},
        "action_correct_pcts": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['action_correct_pcts'].items()},
        "solve_percentages": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['solve_percentages'].items()},
        "diversity_scores": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['diversity_scores'].items()},
        "inference_entropies": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['inference_entropies'].items()},
        "epoch_times": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['epoch_times'].items()}
    },
    "config": CONFIG
}

# Save to JSON file
with open(os.path.join(CONFIG['common_baseline_dir'], 'baseline_metrics.json'), 'w') as f:
    json.dump(metrics_json, f, indent=4)








