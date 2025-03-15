import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile
# from gym_pcgrl.envs.probs.loderunner.engine import chk_playability
from .loderunner_agent_playability import chk_playability
# from .loderunner_playability import is_level_playable



"""
1) More data (lots)
2) smaller receptive field


"""


"""
Generate a LodeRunner level.

Args:
    
"""
class LRProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 32#22
        self._height = 22#12
        self._prob = {"empty": 0.52, "brick": 0.24, "ladder": 0.10, "rope": 0.05, "solid": 0.04, "gold": 0.04, "enemy": 0.009, "player": 0.001}
        self._border_tile = "solid"
        
        self._max_golds = 20
        
        self._rewards = {"player": 1,
            "collected": 3,
            "golds": 1,
            "reachable_tiles": 0.1

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

        self._max_golds = kwargs.get('max_golds', self._max_golds)
        
        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]
                    
                    
    def _run_game(self, map):
        gameCharacters="01234567"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvl = []
        for i in range(len(map)):
            lvl_row = []
            for j in range(len(map[i])):
                string = map[i][j]
                lvl_row.append(int(string_to_char[string]))
            lvl.append(lvl_row)
        lvl = np.asarray(lvl)
    
        collected = chk_playability(lvl, num_cols=self._width, num_rows=self._height)
        return collected
    
    def _run_is_playable(self, map):
        gameCharacters="01234567"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvl = []
        for i in range(len(map)):
            lvl_row = []
            for j in range(len(map[i])):
                string = map[i][j]
                lvl_row.append(int(string_to_char[string]))
            lvl.append(lvl_row)
        lvl = np.asarray(lvl)
    
        is_playable = is_level_playable(lvl)
        return is_playable
    

                    
    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "player": number of player tiles, "golds": number of gold tiles,
        "enemies": number of enemy tiles
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        # collected = self._run_game(map)
        map_stats = {
            "golds": calc_certain_tile(map_locations, ["gold"]),
            "player": calc_certain_tile(map_locations, ["player"]),
            "collected": 0,
            "reachable_tiles": 0,
            "is_playable": False
        }
        if map_stats["player"] == 1 and map_stats["golds"] > 0:
            # is_playable = self._run_is_playable(map)
            game_output = self._run_game(map)
            map_stats["collected"]  = len(game_output[2])
            map_stats["is_playable"]  = True#is_playable
            map_stats["reachable_tiles"]  = len(game_output[-1])

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
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "golds": get_range_reward(new_stats["golds"], old_stats["golds"], 3, self._max_golds),
            "collected": get_range_reward(new_stats["collected"], old_stats["collected"], -np.inf, np.inf),
            "reachable_tiles": get_range_reward(new_stats["reachable_tiles"], old_stats["reachable_tiles"], np.inf, np.inf)
        }
        
        #calculate the total reward
        total_rewards = rewards["player"] * self._rewards["player"] +\
            rewards["golds"] * self._rewards["golds"] +\
            rewards["collected"] * self._rewards["collected"] +\
            rewards["reachable_tiles"] * self._rewards["reachable_tiles"]
        
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
        # return new_stats["golds"] > 3 and new_stats["collected"] == new_stats["golds"] and new_stats["player"]==1
        # return (new_stats["golds"] > 3) and (new_stats["is_playable"] == True) and (new_stats["player"] == 1)
        return (new_stats["golds"] > 3) and (new_stats["player"] == 1) and (new_stats["collected"] == new_stats["golds"])

    
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
            "golds": new_stats["golds"],
            "collected": new_stats["collected"]
        }
    
    def _render_player_agent_trace(self, map, player_agent_trace):
        tiles = self.get_tile_types()
        self._graphics = {}
        for i in range(len(tiles)):
            self._graphics[tiles[i]] = Image.new("RGBA",(self._tile_size,self._tile_size),(0,0,0))

        full_width = len(map[0])+2*self._border_size[0]
        full_height = len(map)+2*self._border_size[1]
        lvl_image = Image.new("RGBA", (full_width*self._tile_size, full_height*self._tile_size), (0,0,0,0))
        for y in range(full_height):
            for x in range(self._border_size[0]):
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile], ((full_width-x-1)*self._tile_size, y*self._tile_size, (full_width-x)*self._tile_size, (y+1)*self._tile_size))
        for x in range(full_width):
            for y in range(self._border_size[1]):
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, (full_height-y-1)*self._tile_size, (x+1)*self._tile_size, (full_height-y)*self._tile_size))
        
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
        
        for trace_tuple in player_agent_trace:
            xy, tile_type = trace_tuple
            y, x = xy
            lvl_image.paste(self._graphics[tile_type], ((x+self._border_size[0])*self._tile_size, (y+self._border_size[1])*self._tile_size, (x+self._border_size[0]+1)*self._tile_size, (y+self._border_size[1]+1)*self._tile_size))
     
        return lvl_image

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map, player_agent_trace=[]):
        if len(player_agent_trace) > 0:
            return self._render_player_agent_trace(map, player_agent_trace)
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