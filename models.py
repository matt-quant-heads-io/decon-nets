import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange
from functools import reduce
from operator import __add__

class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(
            reduce(
                __add__,
                [
                    (k // 2 + (k - 2 * (k // 2)) - 1, k // 2)
                    for k in self.kernel_size[::-1]
                ],
            )
        )

    def forward(self, input):
        return self._conv_forward(
            # self.zero_pad_2d(input.cuda()), self.weight.cuda(), self.bias.cuda()
            self.zero_pad_2d(input), self.weight, self.bias
        )

class ConstructionNetwork(nn.Module):
    """
    Construction Network for repairing noisy states.
    
    This network takes a noisy grid state and predicts the correct tile types.
    """
    def __init__(self, grid_size, num_tile_types):
        """
        Initialize the network.
        
        Args:
            observation_space: The observation space from the environment
            grid_size: Tuple of (height, width) for the grid
            num_tile_types: Number of possible tile types
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_tile_types = num_tile_types
        
        # Convolutional layers for spatial feature extraction
        n_input_channels = self.num_tile_types
        self.cnn = nn.Sequential(
            Conv2dSamePadding(self.num_tile_types, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            Conv2dSamePadding(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            Conv2dSamePadding(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(start_dim=1),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Create a dummy input with the correct shape
            test_input = torch.zeros((1, 22, 22, self.num_tile_types))
            test_input = rearrange(test_input, 'b h w c -> b c h w')
            flattened_size = self.cnn(test_input).shape[-1]
        
        self.fc1 = nn.Linear(flattened_size, num_tile_types)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, num_tile_types, grid_height, grid_width]
            
        Returns:
            Logits for 8 tile types
        """
        # Handle single state input
        # if len(x.shape) == 3:
        #     # Add batch dimension
        #     x = x.unsqueeze(0)
        
        x = rearrange(x, 'b h w c -> b c h w')
        # x = x.permute(0, 3, 1, 2)
        
        # Forward through CNN
        x = self.cnn(x)
        
        # Final fully connected layer
        x = self.fc1(x)
        
        return x
    
    def predict(self, x):
        self.eval()
        x = rearrange(x, 'b h w c -> b c h w')
    # x = x.permute(0, 3, 1, 2)
    
        # Forward through CNN
        x = self.cnn(x)
        
        # Final fully connected layer
        x = self.fc1(x)
        
        return x

    
    def get_entropy(self, x):
    
        """
        Calculate the entropy of the predicted tile distributions
        """
        self.eval()
        # Forward pass to get logits
        logits = self.predict(x)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Calculate entropy: -sum(p * log(p))
        # Add a small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Return the mean entropy across the map
        return entropy



class DestructionNetwork(nn.Module):
    """
    Destruction Network for predicting the entropy of the Construction Network.
    
    This network takes a grid state and predicts the entropy of the Construction
    Network's output distribution for that state.
    """
    def __init__(self, grid_size, num_tile_types):
        """
        Initialize the network.
        
        Args:
            observation_space: The observation space from the environment
            grid_size: Tuple of (height, width) for the grid
            num_tile_types: Number of possible tile types
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_tile_types = num_tile_types
        
        # Convolutional layers for spatial feature extraction
        n_input_channels = self.num_tile_types
        self.cnn = nn.Sequential(
            Conv2dSamePadding(self.num_tile_types, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            Conv2dSamePadding(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            Conv2dSamePadding(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(start_dim=1),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Create a dummy input with the correct shape
            test_input = torch.zeros((1, 22, 22, self.num_tile_types))
            test_input = rearrange(test_input, 'b h w c -> b c h w')
            flattened_size = self.cnn(test_input).shape[-1]
        
        self.fc1 = nn.Linear(flattened_size, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, num_tile_types, grid_height, grid_width]
            
        Returns:
            Logits for 8 tile types
        """
        # Handle single state input
        # if len(x.shape) == 3:
        #     # Add batch dimension
        #     x = x.unsqueeze(0)
        
        x = rearrange(x, 'b h w c -> b c h w')
        # x = x.permute(0, 3, 1, 2)
        
        # Forward through CNN
        x = self.cnn(x)
        
        # Final fully connected layer
        x = self.fc1(x)
        
        return x
    

class DeConNetwork(nn.Module):
    def __init__(self, grid_size, num_tile_types):
        """
        Initialize the network.
        
        Args:
            observation_space: The observation space from the environment
            grid_size: Tuple of (height, width) for the grid
            num_tile_types: Number of possible tile types
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_tile_types = num_tile_types
        self.de = DestructionNetwork(self.grid_size, self.num_tile_types)
        self.con = ConstructionNetwork(self.grid_size, self.num_tile_types)

    def forward_de(self, x):
        return self.de.forward(x)
    
    def forward_con(self, x):
        return self.con.forward(x)
    
    def forward(self, x, which="con"):
        assert which in ['con', 'de', 'both'], "which must be one of ['con', 'de', 'both']"
        if which == 'both':
            return self.forward_de(x), self.forward_con(x), self.get_actual_con_entropy(x)
        elif which == 'con':
            self.de.eval()
            self.con.train()
            return self.forward_con(x)
        
        else:
            self.con.eval()
            self.de.train()
            return self.forward_de(x)
        
    def get_actual_con_entropy(self, x):
        return self.con.get_entropy(x)



        
