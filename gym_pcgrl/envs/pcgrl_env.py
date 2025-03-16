from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import PIL

"""
The PCGRL GYM Environment
"""
class PcgrlEnv(gym.Env):
    """
    The type of supported rendering
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow"):
        print(f"PROBLEMS: {PROBLEMS}")
        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(self._prob._width * self._prob._height), 1)
        self._max_iterations = self._prob._width * self._prob._height*2
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        self.seed()
        self.viewer = None

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, len(self._prob.get_tile_types()))
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, len(self._prob.get_tile_types()))

    """
    Get the number of tile types in the current problem

    Returns:
        int: the number of tile types
    """
    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    """
    Adjust the parameters for the current environment

    Parameters:
        **kwargs: Arbitrary keyword arguments that will be passed to the problem's adjust_param method.
                 Common parameters include:
                 - change_percentage (float): percentage of tiles that can be changed (0-1)
                 - width (int): change the width of the problem level
                 - height (int): change the height of the problem level
                 - probs (dict(string, float)): change the probability of each tile initialization
                 - border_size (tuple(int, int)): change the border size of the level
                 - border_tile (string): change the border tile type
                 - tile_size (int): change the size of each tile in pixels
    """
    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        
        # Pass parameters to problem and representation
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        
        # Update action and observation spaces
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(self._prob._height, self._prob._width))

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int: the used seed (same as input if not None)
    """
    def seed(self, seed=None):
        self._rep.seed(seed)
        seed = self._prob.seed(seed)
        return [seed]

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation
    """
    def reset(self, *, seed=None, options=None, target_map=None):
        if seed is not None:
            self.seed(seed)
            
        self._changes = 0
        self._iteration = 0
        self._rep.reset(self._prob._height, self._prob._width, get_int_prob(self._prob._prob, self._prob.get_tile_types()), target_map=target_map)
        self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros((self._prob._height, self._prob._width))
        
        observation = self._rep.get_observation()
        # observation['heatmap'] = self._heatmap
        info = {}
        
        return observation, info

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile value that can be used for padding
    """
    def get_border_tile(self):
        return self._prob._border_tile

    """
    Update the environment observation based on the action

    Parameters:
        action: the action to take based on the action_space

    Returns:
        observation: the current observation after taking the action
        float: the reward that happened because of the action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information about the current step
    """
    def step(self, action):
        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1.0
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        # calculate the values
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        done = self._prob.get_episode_over(self._rep_stats,old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info(self._rep_stats,old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        info["solved"] = self._prob.get_episode_over(self._rep_stats, old_stats) 
        info["final_map"] = get_string_map(self._rep._map, self._prob.get_tile_types())
        info["x"] = self._rep._x
        info["y"] = self._rep._y
        info["reward"] = reward
        #return the values
        truncated = False
        return observation, reward, done, truncated, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self, mode='human'):
        tile_size=16
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gymnasium.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen

    """
    Close the environment
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


