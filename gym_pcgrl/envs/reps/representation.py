from gymnasium.utils import seeding
from gym_pcgrl.envs.helper import gen_random_map
import random
import numpy as np

"""
The base class of all the representations
"""
class Representation:
    """
    The base constructor where all the representation variable are defined with default values
    """
    def __init__(self):
        self._random_start = True
        self._map = None
        self._old_map = None
        self._random = None
        self._x = 0
        self._y = 0

        self.seed()

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int: the used seed (same as input if not None)
    """
    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        self._random = random
        return seed

    """
    Resets the current representation

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
        target_map (str, optional): path to the target map file
    """
    def reset(self, width, height, prob, target_map=None):
        if target_map:
            # Load the target map from file
            with open(target_map, 'r') as f:
                lines = f.readlines()
            
            # Remove empty lines and strip whitespace
            lines = [line.strip() for line in lines if line.strip()]
            
            # Create numpy array for the map
            h = len(lines)
            w = len(lines[0])
            self._map = np.zeros((h, w), dtype=np.uint8)
            
            # Character to int mapping for Zelda
            char_to_int = {
                '.': 0,  # empty
                'w': 1,  # wall
                'g': 2,  # goal
                '+': 3,  # key
                'A': 4,  # agent
                '1': 5,  # enemy 1
                '2': 6,  # enemy 2
                '3': 7   # enemy 3
            }
            
            # Convert characters to integers
            for i, line in enumerate(lines):
                for j, char in enumerate(line):
                    self._map[i][j] = char_to_int.get(char, 0)
        elif self._random_start:
            self._map = gen_random_map(self._random, width, height, prob)
        else:
            self._map = np.zeros((height, width), dtype=np.uint8)
        self._old_map = self._map.copy()

    """
    Adjust the current used parameters

    Parameters:
        random_start (boolean): if the system will restart with a random map or not
        x (int): the x position of the current tile being modified
        y (int): the y position of the current tile being modified
        map (numpy.ndarray): the current map state
        old_map (numpy.ndarray): the previous map state
    """
    def adjust_param(self, **kwargs):
        self._random_start = kwargs.get('random_start', self._random_start)
        self._x = kwargs.get('x', self._x)
        self._y = kwargs.get('y', self._y)
        
        map = kwargs.get('map')
        if map is not None:
            self._map = map.copy()
            
        old_map = kwargs.get('old_map')
        if old_map is not None:
            self._old_map = old_map.copy()

    """
    Gets the action space used by the representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        ActionSpace: the action space used by that representation
    """
    def get_action_space(self, width, height, num_tiles):
        raise NotImplementedError('get_action_space is not implemented')

    """
    Get the observation space used by the representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        ObservationSpace: the observation space used by that representation
    """
    def get_observation_space(self, width, height, num_tiles):
        raise NotImplementedError('get_observation_space is not implemented')

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment
    """
    def get_observation(self):
        raise NotImplementedError('get_observation is not implemented')

    """
    Update the representation with the current action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        raise NotImplementedError('update is not implemented')

    """
    Modify the level image with any special modification based on the representation

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """
    def render(self, lvl_image, tile_size, border_size):
        return lvl_image
