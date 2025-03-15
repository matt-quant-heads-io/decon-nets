import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile
from gym_pcgrl.envs.probs.loderunner.engine import AStarAgent
from gym_pcgrl.envs.probs.loderunner.tpkldiv import get_tpkldiv
import time

"""
Generate a playable LodeRunner level.
    
"""
class LRProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 32
        self._height = 22
        self._prob ={"empty": 0.55, "brick":0.24, "ladder":0.10, "rope": 0.04, "solid": 0.03, "gold": 0.03, "enemy": 0.005, "player":0.005}
        self._border_tile = "solid"

        self._max_enemies = 5
        self._min_golds = 2
        self._max_golds = 10
        self._target_solution = 80
        
        self._rewards = {
            "golds_collected": 1,
            "golds": 0.5,
            "enemies":0.5,
            "dist_win":0.1,
            "similarity":0.5,
            "sol_length": 0.05
        }
        
        
    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "brick", "ladder", "rope", "solid", "gold", "enemy", "player"]
    
    
    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid"
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)
        self._max_golds = kwargs.get('max_golds', self._max_golds)
        self._min_golds = kwargs.get('min_golds', self._min_golds)
        
        self._target_solution = kwargs.get('min_solution', self._target_solution)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]
                    
    
    """
    Private function that runs the game on the input level

    Parameters:
        map (string[][]): the input level to run the game on

    Returns:
        int: number of golds colleted by the agent
        int: how close you are to winning (0 if you win)
        int: the path length to reach the golds(if all golds are not reachable, 
            path length to reachable golds)
    """
    def _run_game(self, map):
        gameCharacters=".b#-BGEM"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvl = []
        for i in range(len(map)):
            lvl_row = []
            for j in range(len(map[i])):
                string = map[i][j]
                lvl_row.append(string_to_char[string])
            lvl.append(lvl_row)
    
        colleted_golds,dist_win,path_len = AStarAgent(lvl)
        return colleted_golds,dist_win,path_len
    
    
    """
    Get the similarity of the current level with target level(s)

    Parameters:
        map (string[][]): the input level to run the game on

    Returns:
        float: tpkl-div score
    """
    def _get_similarity(self, map):
        gameCharacters=".b#-BGEM"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvl = []
        for i in range(len(map)):
            lvl_row = []
            for j in range(len(map[i])):
                string = map[i][j]
                lvl_row.append(string_to_char[string])
            lvl.append(lvl_row)
    
        sim = get_tpkldiv(lvl)
        return sim
                    
                    
    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "player": number of player tiles, "golds": number of gold tiles,
        "enemies": number of enemy tiles
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "golds": calc_certain_tile(map_locations, ["gold"]),
            "enemies": calc_certain_tile(map_locations, ["enemy"]),
            "golds_collected": 0,
            "dist_win":10,
            "similarity": self._get_similarity(map),
            "sol_length": 0
        }
        if map_stats["player"] == 1 and map_stats["golds"] > self._min_golds-1:
            map_stats["golds_collected"],map_stats["dist_win"],map_stats["sol_length"] = self._run_game(map)
        return map_stats
        
    
    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and higher number of collected gold is rewarded
        rewards = {
            "golds_collected": get_range_reward(new_stats["golds_collected"],old_stats["golds_collected"], self._max_golds, self._max_golds),
            "golds": get_range_reward(new_stats["golds"], old_stats["golds"], self._min_golds, self._max_golds),
            "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], 1, self._max_enemies),
            "dist_win": get_range_reward(new_stats["dist_win"], old_stats["dist_win"], -np.inf, -np.inf),
            "similarity": new_stats["similarity"],
            "sol_length": get_range_reward(new_stats["sol_length"], old_stats["sol_length"], np.inf, np.inf)
        }
        
        #calculate the total reward
        total_rewards = rewards["golds_collected"] * self._rewards["golds_collected"] +\
            rewards["golds"] * self._rewards["golds"] +\
            rewards["enemies"] * self._rewards["enemies"] +\
            rewards["dist_win"] * self._rewards["dist_win"] +\
            rewards["sol_length"] * self._rewards["sol_length"]
        
        return total_rewards           
            
    
    
    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats):
        return new_stats["golds"] >= self._min_golds and new_stats["golds"] < self._max_golds and new_stats["golds_collected"] == new_stats["golds"] and new_stats["sol_length"] >= self._target_solution 
    
    
    """
    ** This function is called at the end of the episode **    
    Get the similarity score of the current level with target levels
   
    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action

    Returns:
        float: The current similarity score
    """
    def get_similarity_rewrad(self,new_stats):
        return new_stats["similarity"] * self._rewards["similarity"]
    
    
    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "golds_collected": new_stats["golds_collected"],
            "golds": new_stats["golds"],
            "enemies": new_stats["enemies"],
            "dist_win": new_stats["dist_win"],
            "similarity": new_stats["similarity"],
            "sol_length": new_stats["sol_length"]
        }
    
     
    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/loderunner/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/loderunner/solid.png").convert('RGBA'),
                "brick": Image.open(os.path.dirname(__file__) + "/loderunner/brick.png").convert('RGBA'),
                "ladder": Image.open(os.path.dirname(__file__) + "/loderunner/ladder.png").convert('RGBA'),
                "rope": Image.open(os.path.dirname(__file__) + "/loderunner/rope.png").convert('RGBA'),
                "gold": Image.open(os.path.dirname(__file__) + "/loderunner/gold.png").convert('RGBA'),
                "enemy": Image.open(os.path.dirname(__file__) + "/loderunner/enemy.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/loderunner/player.png").convert('RGBA'),
            }
        return super().render(map)
    
