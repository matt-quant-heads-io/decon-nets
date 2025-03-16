import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from typing import List

@dataclass
class NoiseStep:
    state_before: np.ndarray
    x_pos: int
    y_pos: int
    noise_action: int
    state_after: np.ndarray

class PCGRLNoiseWrapper(gym.Wrapper):
    """
    Wrapper for PCGRL environment to add controlled noise.
    
    This wrapper adds noise to a given percentage of tiles in the PCGRL environment.
    It also keeps track of the clean state for evaluation purposes.
    """
    def __init__(self, env, difficulty_pct=0.05):
        """
        Initialize the wrapper.
        
        Args:
            env: The PCGRL environment to wrap
            difficulty_pct: Percentage of tiles to corrupt (0.0 to 1.0)
        """
        super().__init__(env)
        self.difficulty_pct = difficulty_pct
        self.clean_state = None
        self.x, self.y = None, None
        self.grid_size = None
        self.num_tile_types = None
        self.current_trajectory: List[NoiseStep] = []
        
    def reset(self):
        """
        Reset the environment and add controlled noise.
        
        Returns:
            Noisy observation
        """
        # Reset the environment to get a clean state
        self.clean_state_tuple, info = self.env.unwrapped.reset()
        
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
    
    def _add_noise(self, state):
        """
        Add noise to observation based on difficulty percentage.
        
        Args:
            state: Clean state to add noise to
            
        Returns:
            Noisy state
        """
        # Create a copy to avoid modifying the original state
        noisy_state = state.copy()
        
        # Calculate number of tiles to corrupt
        total_tiles = np.prod(self.grid_size)
        num_noisy_tiles = int(total_tiles * self.difficulty_pct)
        noisy_tiles_applied = 0
        
        # Randomly select positions to corrupt
        flat_indices = np.random.choice(total_tiles, num_noisy_tiles, replace=False)
        
        # Convert flat indices to grid positions
        x_indices = flat_indices // self.grid_size[1]
        y_indices = flat_indices % self.grid_size[1]
        
        # Apply random noise to selected positions
        while noisy_tiles_applied <=  num_noisy_tiles:
            # Store state before noise
            state_before = noisy_state.copy()
            
            # Get current position from environment representation
            x_pos, y_pos = self.env.unwrapped._rep._x, self.env.unwrapped._rep._y
            
            # Generate and apply noise
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
                state_after=noisy_state.copy()
            ))
            noisy_tiles_applied += 1
        
        return noisy_state
    
    def step(self, action, position=None):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            position: Optional position to apply action (for DeCon experiments)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if position is not None:
            # If position is provided, use it (for DeCon experiments)
            return self.env.unwrapped.step((position, action))
        else:
            # Otherwise, use standard step
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