import os
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import wandb
from tqdm import tqdm
from models import ConstructionNetwork, DestructionNetwork
from pcgrl_wrappers import PCGRLNoiseWrapper
from utils import calculate_entropy, evaluate_solvability, evaluate_diversity
import torch.nn.functional as F
from inference import infer
import time
import json
import copy
from models import DeConNetwork

import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from typing import List


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
    'difficulty_levels': [0.60, 0.80, 1.00],
    
    # Networks
    'grid_size': (7, 11),  # Example size, adjust based on your PCGRL env
    'crop_size': (22, 22),  # Size of observation window
    'num_tile_types': 8,
    'learning_rate_construction': 1e-4,
    'learning_rate_destruction': 1e-4,
    
     # Training
    'num_state_action_pairs_per_difficulty': 1000,
    'evaluate_every_num_epochs': 40,
    'entropy_threshold': 0.03,
    'num_epochs_per_difficulty': 120,
    'batch_size': 128,
    'inference_trials': 100,
    'initial_dataset_size': 100,  # For initial destruction network training
    'inference_entropy_thresh': 0.5,
    'solve_pct_thresh': 0.5,
    'patience': 100, # Number of atempts the agent is given to solve the constraints for given difficulty

    # Directories
    'warmup_dir': 'warmup',
    'random_agent_dir': 'random_agent_experiments',
    'common_baseline_dir': 'common_baseline', # NOTE: This contains the starting checkpoints for running the random and decon experiments
    'baseline_dir': 'baseline',
    'decon_dir': 'decon_experiments',
}

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

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, device=None):
        self.states = []
        self.actions = []
        self.entropies = []
        
        for trajectory in trajectories:
            for traj in trajectory:
                # import pdb; pdb.set_trace()
                self.states.append(traj['state'])
                self.actions.append(traj['action'])
                self.entropies.append(traj['entropy'])
        
        self.states = torch.stack(self.states).to(device)
        self.actions = torch.stack(self.actions).to(device)
        self.entropies = torch.stack(self.entropies).float().to(device)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.entropies[idx]

@dataclass
class NoiseStep:
    state_before: np.ndarray
    x_pos: int
    y_pos: int
    noise_action: int
    state_after: np.ndarray
    repair_action: int

class PCGRLNoiseWrapper(gym.Wrapper):
    """
    Wrapper for PCGRL environment to add controlled noise.
    
    This wrapper adds noise to a given percentage of tiles in the PCGRL environment.
    It also keeps track of the clean state for evaluation purposes.
    """
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: The PCGRL environment to wrap
            difficulty_pct: Percentage of tiles to corrupt (0.0 to 1.0)
        """
        super().__init__(env)
        self.clean_state = None
        self.x, self.y = None, None
        self.grid_size = None
        self.num_tile_types = None
        self.current_trajectory: List[NoiseStep] = []
        self.env = env
        
    def reset(self, target_map):
        """
        Reset the environment and add controlled noise.
        
        Returns:
            Noisy observation
        """
        # Reset the environment to get a clean state
        self.clean_state_tuple, info = self.env.unwrapped.reset(target_map=target_map)
        
        # Determine grid size and number of tile types
        # import pdb; pdb.set_trace()
        self.x, self.y = self.clean_state_tuple['pos']
        self.clean_state = self.clean_state_tuple['map']
        if self.grid_size is None:
            if isinstance(self.clean_state, np.ndarray):
                self.grid_size = self.clean_state.shape[:2]  # Assuming shape is (H, W, C)
                self.num_tile_types = self.clean_state.shape[-1]  # Last dimension is tile types
            else:
                # Handle other observation types if needed
                raise ValueError("Unsupported observation type")
        
        # Add noise to the clean state
        noisy_state = self._add_noise(self.clean_state)
        
        return noisy_state
    
    def _add_noise(self, noise_pct, noise_fn=None):
        """
        Add noise to observation based on difficulty percentage.
        
        Args:
            state: Clean state to add noise to
            
        Returns:
            Noisy state
        """
        # Create a copy to avoid modifying the original state
        state = self.env.unwrapped._map["map"]
        noisy_state = copy.deepcopy(state)
        
        # Calculate number of tiles to corrupt
        total_tiles = np.prod(self.grid_size)
        num_noisy_tiles = int(total_tiles * noise_pct)
        noisy_tiles_applied = 0
        
        # Apply random noise to selected positions
        while noisy_tiles_applied <=  num_noisy_tiles:
            # Store state before noise
            state_before = copy.deepcopy(noisy_state.copy())
            
            # Get current position from environment representation
            x_pos, y_pos = self.env.unwrapped._rep._x, self.env.unwrapped._rep._y
            repair_action = state_before[y_pos][x_pos]
            
            # Generate and apply noise
            if noise_fn:
                noise_action = noise_fn(state_before)
            else:
                noise_action = np.random.randint(0, self.num_tile_types-1)
            # random_tile = np.zeros(self.num_tile_types)
            # random_tile[noise_action] = 1
            print(noisy_state.shape)
            noisy_state[y_pos, x_pos] = noise_action
            
            # Store step in trajectory
            self.current_trajectory.append(NoiseStep(
                state_before=state_before,
                x_pos=x_pos,
                y_pos=y_pos,
                noise_action=noise_action,
                state_after=copy.deepcopy(noisy_state),
                repair_action=repair_action
            ))
            noisy_tiles_applied += 1
        
        return noisy_state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            position: Optional position to apply action (for DeCon experiments)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        
        return self.env.unwrapped.step(action)
    
    def get_clean_state(self):
        """
        Get the clean state before noise was applied.
        
        Returns:
            Clean state
        """
        return self.clean_state

    def get_trajectory(self):
        """Return the current noise application trajectory."""
        return self.current_trajectory


class RandomTrainer:
    def __init__(self, env, grid_size, num_tile_types, device):
        self.env = env
        self.grid_size = grid_size
        self.num_tile_types = num_tile_types
        
        
        # Initialize networks
        self.con = ConstructionNetwork(grid_size, num_tile_types).to(device)
        
        # Initialize optimizers
        self.con_optimizer = Adam(
            self.con.parameters(),
            lr=CONFIG['learning_rate_construction']
        )
        self.con_loss = nn.CrossEntropyLoss()
        
        # Initialize wandb
        # wandb.init(project=CONFIG['wandb_project'])
        
        # Create directories
        os.makedirs(CONFIG['random_agent_dir'], exist_ok=True)

    def transform(self, obs, x, y, obs_size):
        map = obs
        size = obs_size[0]
        pad = obs_size[0] // 2
        padded = np.pad(map, pad, constant_values=1)
        cropped = padded[y: y + size, x: x + size]
        
        return cropped

    
    def save_checkpoint(self, difficulty_pct):
        """Save model checkpoints."""
        checkpoint = {
            'construction_net_state_dict': self.con.state_dict(),
            'construction_optimizer_state_dict': self.con_optimizer.state_dict(),
            'difficulty': difficulty_pct
        }
        torch.save(checkpoint, os.path.join(CONFIG['random_agent_dir'], f'decon_difficulty_{difficulty_pct:.2f}.pt'))

    def generate_random_trajectory(self, difficulty, grid_size):
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
        state_pos_dict, info = self.env.unwrapped.reset(target_map="./goal_maps/zelda/1.txt")
        y, x = state_pos_dict['pos'] 
        y, x = y.item(), x.item()
        state = state_pos_dict['map'] 
        
        for _ in range(trajectory_length):
            step_dict = {}
            step_dict['state'] = F.one_hot(torch.tensor(self.transform(state, x, y, CONFIG['crop_size'])).long(), num_classes=8).float().unsqueeze(0).cuda()
            step_dict['x'], step_dict['y'] = x, y   

            # Sample a random action from the environment's action space
            # NOTE: here is the random sample
            action = self.env.unwrapped.action_space.sample()
            # step_dict['action'] = action.item()
            
            # Take the action in the environment
            next_state_pos_dict, reward, done, truncated, info = self.env.unwrapped.step(action)

            step_dict['reward'] = reward
            step_dict['done'] = done
            
            entropy = self.con.get_entropy(F.one_hot(torch.tensor(self.transform(state, x, y, CONFIG['crop_size'])).long(), num_classes=8).float().unsqueeze(0).cuda())
            next_state = next_state_pos_dict['map']
            y, x = next_state_pos_dict['pos']
             
            step_dict['action'] = F.one_hot(torch.tensor([action.item()]), num_classes=8).float()
            step_dict['entropy'] = torch.tensor([entropy]).float()
            x, y, = x.item(), y.item()        
            # Store the transition
            trajectory.append(step_dict)
            
            # Update state for next iteration
            state = next_state
                
        return trajectory
    
    def generate_decon_trajectory(self, difficulty, grid_size):
        """
        Generate a trajectory from an environment object using a trained neural network
        that selects actions based on predicted state entropy.
        
        Args:
            env: The environment object to generate the trajectory from
            difficulty (float): Difficulty factor that affects trajectory length
            grid_size (int): Base grid size for calculating trajectory length
            trained_network: A pre-trained construction network for entropy prediction
            
        Returns:
            tuple: (list of state, action, reward, next_state, done dictionaries, float average entropy)
        """
        trajectory = []
        trajectory_length = int(difficulty * grid_size[0] * grid_size[1])
        
        # Reset the environment to get initial state
        state_pos_dict, info = self.env.unwrapped.reset(target_map="./goal_maps/zelda/1.txt")
        y, x = state_pos_dict['pos'] 
        y, x = y.item(), x.item()
        state = state_pos_dict['map'] 
        
        # Track entropies for averaging
        trajectory_entropies = []
        
        for _ in range(trajectory_length):
            step_dict = {}
            step_dict['state'] = state
            step_dict['x'], step_dict['y'] = x, y   

            # Use the trained network to select the action with highest predicted entropy
            action_entropies = []
            possible_actions = range(self.env.unwrapped.action_space.n)  # Assuming Discrete action space
            
            # For each possible action, predict the entropy of the resulting state
            for a in possible_actions:
                # Create a copy of the environment to simulate the action
                
                next_state = copy.deepcopy(state)
                next_state[y, x] = a  # Apply the action at the current x,y position
                
                # Convert state to network input format
                state_tensor = F.one_hot(torch.tensor(self.transform(next_state, x, y, CONFIG['crop_size'])).long(), num_classes=8).float().unsqueeze(0).cuda()
                entropy = self.con.get_entropy(state_tensor)
                action_entropies.append(entropy.item())
            
            # Select the action with the highest predicted entropy
            action = possible_actions[action_entropies.index(max(action_entropies))]
            step_dict['action'] = action
            
            # Store the entropy of the selected action
            step_dict['entropy'] = max(action_entropies)
            trajectory_entropies.append(max(action_entropies))
            
            # Take the selected action in the environment
            next_state_pos_dict, reward, done, truncated, info = self.env.unwrapped.step(action)

            step_dict['reward'] = reward
            step_dict['done'] = done
            next_state = next_state_pos_dict['map']
            y, x = next_state_pos_dict['pos']
            x, y, = x.item(), y.item()        
            # Store the transition
            trajectory.append({
                "x": x,
                "y": y,
                "state": state_tensor,
                "action": F.one_hot(torch.tensor([step_dict['action']]), num_classes=8).float(),
                "entropy": torch.tensor([step_dict['entropy']]).float()
            })
            
            # Update state for next iteration
            state = next_state
        
        # Calculate average entropy for the trajectory
        avg_entropy = sum(trajectory_entropies) / len(trajectory_entropies) if trajectory_entropies else 0.0
                
        return trajectory, avg_entropy

    
    def train(self):
        state_pos_dict, info = self.env.unwrapped.reset(target_map="./goal_maps/zelda/1.txt")
        y, x = state_pos_dict['pos'] 
        y, x = y.item(), x.item()
        state = state_pos_dict['map'] 
        solve_pct = None
        mean_entropy = None

        metrics_per_difficulty = {
            'con_losses': {},
            'entropies': {},
            'action_correct_pcts': {},
            'solve_percentages': {},
            'diversity_scores': {},
            'inference_entropies': {},
            'epoch_times': {}  # Add timing metrics
        }


        for difficulty in CONFIG['difficulty_levels']:
            solve_pct = -np.inf
            mean_entropy = np.inf
            total_state_action_pairs_per_difficulty = 0
            curr_patience = 0
            while solve_pct < CONFIG['solve_pct_thresh'] and mean_entropy > CONFIG['inference_entropy_thresh'] and curr_patience <= CONFIG['patience']:
                curr_patience += 1
            
                trajectories = []
                num_state_action_pairs = 0
                while num_state_action_pairs <= CONFIG['num_state_action_pairs_per_difficulty']:
                    trajectory = self.generate_random_trajectory(difficulty, CONFIG['grid_size'])
                    num_state_action_pairs += len(trajectory)
                    trajectories.append(trajectory)


                dataset = TrajectoryDataset(trajectories, device='cuda')
                dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

                total_state_action_pairs_per_difficulty += len(trajectories)
                # Lists to store metrics for current difficulty
                entropies = []
                action_correct_pcts = []
                solve_percentages = []  # Add list for solve percentages
                diversity_scores = []   # Add list for diversity scores
                mean_entropies = []
                epoch_times = []  # Add list for epoch times
                con_losses = []

                # Training loop
                for epoch in range(CONFIG['num_epochs_per_difficulty']):
                    epoch_start_time = time.time()
                    epoch_loss = 0
                    epoch_entropy = 0
                    num_batches = 0
                    
                    num_correct_per_epoch = 0
                    batches_per_epoch = 0
                    
                    for batch_states, batch_actions, batch_entropies in dataloader:
                        batch_states = batch_states.cuda()
                        batch_actions = batch_actions.cuda()
                        batch_entropies = batch_entropies.cuda()

                        self.con_optimizer.zero_grad(set_to_none=True)

                        pred_actions = self.con(batch_states.squeeze(1))                    
                        
                        
                        loss_con = self.con_loss(pred_actions, batch_actions.squeeze(1))

                        loss_con.backward()
                        
                        
                        
                        self.con_optimizer.step()

                        num_correct_per_epoch += (torch.max(pred_actions, dim=1).indices == torch.max(torch.squeeze(batch_actions, 1), dim=1).indices).type(torch.float).sum().item()

                        epoch_entropy += batch_entropies.sum()                        
                        num_batches += len(batch_states)


                    # Average metrics for the epoch
                    avg_loss_con = loss_con.mean().item()
                    avg_entropy = epoch_entropy.item() / num_batches
                    avg_correct_pct = num_correct_per_epoch / num_batches
                    con_losses.append(avg_loss_con)
                    entropies.append(avg_entropy)
                    action_correct_pcts.append(avg_correct_pct)

                    # Calculate epoch time
                    epoch_time = time.time() - epoch_start_time
                    epoch_times.append(epoch_time)
                    
                    print(f'Epoch [{epoch+1}/{CONFIG["num_epochs_per_difficulty"]}], Con Loss: {avg_loss_con:.4f}, Entropy: {avg_entropy:.4f}, Correct {avg_correct_pct:.4f}, Time: {epoch_time:.2f}s')
                    
                    # Every 100 epochs, evaluate the network's performance
                    if (epoch + 1) % CONFIG['evaluate_every_num_epochs'] == 0:
                        construction_net_copy = copy.deepcopy(self.con)
                        
                        wrapped_network = ConstructionNetworkWrapper(construction_net_copy)
                        solve_pct, div_score, mean_entropy = infer(
                            wrapped_network,
                            self.env,
                            trials=CONFIG['inference_trials'],  # Reduced number of trials for faster evaluation during training
                            verbose=False,
                            div_cutoff=0.10,
                            goal_map_path="./goal_maps/zelda/1.txt"
                        )

                        self.save_checkpoint(difficulty_pct=difficulty)

                        
                        solve_percentages.append(solve_pct)
                        diversity_scores.append(div_score)
                        mean_entropies.append(mean_entropy)
                        print(f'Evaluation - Solve %: {solve_pct:.2f}, Diversity: {div_score:.4f}, Entropy: {mean_entropy:.4f}')

                        # STEP 1: Train both the de and the con simultaneously for difficulty 0.6 via randomly sampled noise



                        
                        #         Run inference with the con for 100 trials
                        
                        # STEP 2: Generate trajectory from de-sampled noise on difficulty 0.8 --> 

                        #         Run inference for 100 trials, if solvability >= 50% stop record the state-action pairs used in training for difficulty 0.8
                        #                                       if solvability < 50% restart STEP 2 again

                        # For each iteration for STEP 2 --> record the con and de loss curves, training entropies, average inference results, and the total number of state-action pairs used to train (until threshold was met)


                        # STEP 3: Generate trajectory from de-sampled noise on difficulty 1.0 --> 

                        #         Run inference for 100 trials, if solvability >= 50% stop record the state-action pairs used in training for difficulty 0.8
                        #                                       if solvability < 50% restart STEP 3 again
                    
                        # For each iteration for STEP 3 --> record the con and de loss curves, training entropies, average inference results, and the total number of state-action pairs used to train (until threshold was met)

                        # Store the plots in json
            # Store metrics for this difficulty level
            metrics_per_difficulty['con_losses'][difficulty] = con_losses
            metrics_per_difficulty['entropies'][difficulty] = entropies
            metrics_per_difficulty['action_correct_pcts'][difficulty] = action_correct_pcts
            metrics_per_difficulty['solve_percentages'][difficulty] = solve_percentages
            metrics_per_difficulty['diversity_scores'][difficulty] = diversity_scores
            metrics_per_difficulty['inference_entropies'][difficulty] = mean_entropies
            metrics_per_difficulty['epoch_times'][difficulty] = epoch_times

        # Create plots for metrics
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        for difficulty in CONFIG['difficulty_levels']:
            plt.plot(con_losses, label=f'Con Loss (diff={difficulty})')
        plt.title('Construction Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot entropies
        plt.subplot(2, 2, 2)
        for difficulty in CONFIG['difficulty_levels']:
            plt.plot(metrics_per_difficulty['entropies'][difficulty], label=f'Diff={difficulty}')
        plt.title('Training Entropies')
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.legend()
        
        # Plot solve percentages
        plt.subplot(2, 2, 3)
        for difficulty in CONFIG['difficulty_levels']:
            plt.plot(metrics_per_difficulty['solve_percentages'][difficulty], label=f'Diff={difficulty}')
        plt.title('Solve Percentages')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Solve %')
        plt.legend()
        
        # Plot diversity scores
        plt.subplot(2, 2, 4)
        for difficulty in CONFIG['difficulty_levels']:
            plt.plot(metrics_per_difficulty['diversity_scores'][difficulty], label=f'Diff={difficulty}')
        plt.title('Diversity Scores')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Diversity Score')
        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['random_agent_dir'], 'decon_curriculum_metrics.png'))
        plt.close()

        # Save metrics data to JSON file
        metrics_json = {
            "metrics_per_difficulty": {
                "con_losses": {str(k): [float(x) for x in v] for k, v in metrics_per_difficulty['con_losses'].items()},
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
        with open(os.path.join(CONFIG['random_agent_dir'], 'random_agent_metrics.json'), 'w') as f:
            json.dump(metrics_json, f, indent=4)

        



env = gym.make(f"{CONFIG['pcgrl_game']}-{CONFIG['pcgrl_representation']}-v0")
env = PCGRLNoiseWrapper(env)           

RandomTrainer(env, CONFIG['grid_size'], CONFIG['num_tile_types'], 'cuda').train()