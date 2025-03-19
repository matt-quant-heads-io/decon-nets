import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb
from tqdm import tqdm

from models import ConstructionNetwork, DestructionNetwork
from pcgrl_wrappers import PCGRLNoiseWrapper
from .src.config import CONFIG


class DeConTrainer:
    def __init__(self, env, grid_size, num_tile_types, device):
        self.env = env
        self.grid_size = grid_size
        self.num_tile_types = num_tile_types
        
        # Initialize networks
        self.construction_net = ConstructionNetwork(grid_size, num_tile_types).to(CONFIG['device'])
        self.destruction_net = DestructionNetwork(grid_size, num_tile_types).to(CONFIG['device'])
        
        # Initialize optimizers
        self.construction_optimizer = Adam(
            self.construction_net.parameters(),
            lr=CONFIG['learning_rate_construction']
        )
        self.destruction_optimizer = Adam(
            self.destruction_net.parameters(),
            lr=CONFIG['learning_rate_destruction']
        )
        
        # Initialize wandb
        wandb.init(project=CONFIG['wandb_project'])
        
        # Create directories
        os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
        os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
        
    def train_construction_network(self, difficulty_pct):
        """Train the construction network via supervised learning."""
        self.construction_net.train()
        self.destruction_net.eval()
        
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for episode in range(CONFIG['num_episodes_per_difficulty']):
            obs = self.env.reset()
            episode_loss = 0
            
            for step in range(CONFIG['max_steps_per_episode']):
                # Get original observation for supervision
                original_obs = self.env.get_original_obs()
                
                # Convert observations to tensors
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(CONFIG['device'])
                original_tensor = torch.FloatTensor(original_obs).unsqueeze(0).to(CONFIG['device'])
                
                # Forward pass
                logits = self.construction_net(obs_tensor)
                loss = torch.nn.functional.cross_entropy(logits, original_tensor.argmax(dim=-1))
                
                # Backward pass
                self.construction_optimizer.zero_grad()
                loss.backward()
                self.construction_optimizer.step()
                
                episode_loss += loss.item()
                
                # Calculate accuracy
                predictions = logits.argmax(dim=-1)
                correct_predictions += (predictions == original_tensor.argmax(dim=-1)).sum().item()
                total_predictions += predictions.numel()
                
                # Step environment
                obs, _, done, _ = self.env.step(predictions[0].item())
                if done:
                    break
            
            total_loss += episode_loss
            
            if episode % CONFIG['log_interval'] == 0:
                wandb.log({
                    'construction_loss': episode_loss,
                    'construction_accuracy': correct_predictions / total_predictions,
                    'difficulty': difficulty_pct
                })
        
        return total_loss / CONFIG['num_episodes_per_difficulty']
    
    def train_destruction_network(self, difficulty_pct):
        """Train the destruction network to predict construction network's entropy."""
        self.construction_net.eval()
        self.destruction_net.train()
        
        total_loss = 0
        
        for episode in range(CONFIG['num_episodes_per_difficulty']):
            obs = self.env.reset()
            episode_loss = 0
            
            for step in range(CONFIG['max_steps_per_episode']):
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(CONFIG['device'])
                
                # Get actual entropy from construction network
                with torch.no_grad():
                    actual_entropy = self.construction_net.get_entropy(obs_tensor)
                
                # Get predicted entropy
                predicted_entropy = self.destruction_net(obs_tensor)
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(predicted_entropy, actual_entropy)
                
                # Backward pass
                self.destruction_optimizer.zero_grad()
                loss.backward()
                self.destruction_optimizer.step()
                
                episode_loss += loss.item()
                
                # Step environment
                obs, _, done, _ = self.env.step(0)  # Dummy action
                if done:
                    break
            
            total_loss += episode_loss
            
            if episode % CONFIG['log_interval'] == 0:
                wandb.log({
                    'destruction_loss': episode_loss,
                    'difficulty': difficulty_pct
                })
        
        return total_loss / CONFIG['num_episodes_per_difficulty']
    
    def evaluate_entropy(self):
        """Evaluate the mean entropy of construction network on validation environments."""
        self.construction_net.eval()
        total_entropy = 0
        num_steps = 0
        
        for episode in range(CONFIG['evaluation_episodes']):
            obs = self.env.reset()
            
            for step in range(CONFIG['max_steps_per_episode']):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(CONFIG['device'])
                entropy = self.construction_net.get_entropy(obs_tensor)
                total_entropy += entropy.item()
                num_steps += 1
                
                obs, _, done, _ = self.env.step(0)  # Dummy action
                if done:
                    break
        
        return total_entropy / num_steps
    
    def save_checkpoint(self, difficulty_pct):
        """Save model checkpoints."""
        checkpoint = {
            'construction_net_state_dict': self.construction_net.state_dict(),
            'destruction_net_state_dict': self.destruction_net.state_dict(),
            'construction_optimizer_state_dict': self.construction_optimizer.state_dict(),
            'destruction_optimizer_state_dict': self.destruction_optimizer.state_dict(),
            'difficulty': difficulty_pct
        }
        torch.save(checkpoint, os.path.join(CONFIG['checkpoint_dir'], f'checkpoint_{difficulty_pct:.2f}.pt'))
    
    def train(self):
        """Main training loop implementing curriculum learning."""
        current_difficulty = CONFIG['initial_difficulty']
        best_entropy = float('inf')
        patience_counter = 0
        
        while current_difficulty <= CONFIG['max_difficulty']:
            print(f"\nTraining at difficulty level: {current_difficulty:.2f}")
            
            # Phase 1: Train Construction Network
            print("Phase 1: Training Construction Network")
            construction_loss = self.train_construction_network(current_difficulty)
            
            # Phase 2: Train Destruction Network
            print("Phase 2: Training Destruction Network")
            destruction_loss = self.train_destruction_network(current_difficulty)
            
            # Evaluate current performance
            mean_entropy = self.evaluate_entropy()
            print(f"Mean entropy: {mean_entropy:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(current_difficulty)
            
            # Update difficulty based on performance
            if mean_entropy < CONFIG['entropy_threshold']:
                if mean_entropy < best_entropy - CONFIG['min_delta']:
                    best_entropy = mean_entropy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= CONFIG['patience']:
                    current_difficulty += CONFIG['difficulty_increment']
                    patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['patience']:
                    print("Training failed to converge at current difficulty level")
                    break
            
            wandb.log({
                'current_difficulty': current_difficulty,
                'mean_entropy': mean_entropy,
                'construction_loss': construction_loss,
                'destruction_loss': destruction_loss
            })
        
        # Save final models
        torch.save(self.construction_net.state_dict(), 
                  os.path.join(CONFIG['model_save_dir'], 'construction_net_final.pt'))
        torch.save(self.destruction_net.state_dict(), 
                  os.path.join(CONFIG['model_save_dir'], 'destruction_net_final.pt'))
        
        wandb.finish() 