from gymnasium.utils import seeding
from PIL import Image

"""
The base class for all the problems that can be handled by the interface
"""
class Problem:
    """
    Constructor for the problem that initialize all the basic parameters
    """
    def __init__(self):
        self._width = 9
        self._height = 9
        tiles = self.get_tile_types()
        self._prob = []
        for _ in range(len(tiles)):
            self._prob.append(1.0/len(tiles))

        self._border_size = (1,1)
        self._border_tile = tiles[0]
        self._tile_size=16
        self._graphics = None

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
        return seed

    """
    Resets the problem to the initial state

    Parameters:
        stats (dictionary): the stats of the current map
    """
    def reset(self, stats):
        self._start_stats = stats

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        raise NotImplementedError('get_tile_types is not implemented')

    """
    Get a list of all the different tile values

    Returns:
        string[]: that contains all the tile values
    """
    def get_tile_values(self):
        raise NotImplementedError('get_tile_values is not implemented')

    """
    Get the current stats of the map

    Returns:
        dict(string,any): the current stats of the map
    """
    def get_stats(self, map):
        raise NotImplementedError('get_stats is not implemented')

    """
    Get the current reward based on the previous stats and the current ones

    Parameters:
        old_stats (dict(string,any)): the old stats of the map
        new_stats (dict(string,any)): the new stats of the map
        changes (int): the number of tiles that have been changed
        heatmap (int[][]): the number of times each tile has been modified

    Returns:
        float: the current reward
    """
    def get_reward(self, new_stats, old_stats, changes, heatmap):
        raise NotImplementedError('get_reward is not implemented')

    """
    Uses the stats to check if the problem ended (episode_over)

    Parameters:
        stats (dict(string,any)): the current stats of the map
        changes (int): the number of tiles that have been changed
        heatmap (int[][]): the number of times each tile has been modified

    Returns:
        boolean: True if the level is over (episode_over)
    """
    def get_episode_over(self, stats, changes, heatmap):
        raise NotImplementedError('get_episode_over is not implemented')

    """
    Get any debug information need to be printed

    Parameters:
        stats (dict(string,any)): the current stats of the map
        changes (int): the number of tiles that have been changed
        heatmap (int[][]): the number of times each tile has been modified

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, stats, changes, heatmap):
        raise NotImplementedError('get_debug_info is not implemented')

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the problem
        graphics or default grey scale colors
    """
    def render(self, map):
        tiles = self.get_tile_types()
        if self._graphics == None:
            self._graphics = {}
            for i in range(len(tiles)):
                color = (i*255/len(tiles),i*255/len(tiles),i*255/len(tiles),255)
                self._graphics[tiles[i]] = Image.new("RGBA",(self._tile_size,self._tile_size),color)

        full_width = len(map[0])+2*self._border_size[0]
        full_height = len(map)+2*self._border_size[1]
        lvl_image = Image.new("RGBA", (full_width*self._tile_size, full_height*self._tile_size), (0,0,0,255))
        for y in range(full_height):
            for x in range(self._border_size[0]):
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile], ((full_width-x-1)*self._tile_size, y*self._tile_size, (full_width-x)*self._tile_size, (y+1)*self._tile_size))
        for x in range(full_width):
            for y in range(self._border_size[1]):
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, (full_height-y-1)*self._tile_size, (x+1)*self._tile_size, (full_height-y)*self._tile_size))
        for y in range(len(map)):
            for x in range(len(map[y])):
                lvl_image.paste(self._graphics[map[y][x]], ((x+self._border_size[0])*self._tile_size, (y+self._border_size[1])*self._tile_size, (x+self._border_size[0]+1)*self._tile_size, (y+self._border_size[1]+1)*self._tile_size))
        return lvl_image

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile initialization
        border_size (tuple(int, int)): change the border size of the level
        border_tile (string): change the border tile type
        tile_size (int): change the size of each tile in pixels
    """
    def adjust_param(self, **kwargs):
        self._width = kwargs.get('width', self._width)
        self._height = kwargs.get('height', self._height)
        
        probs = kwargs.get('probs')
        if probs is not None:
            self._prob = probs
            
        self._border_size = kwargs.get('border_size', self._border_size)
        self._border_tile = kwargs.get('border_tile', self._border_tile)
        self._tile_size = kwargs.get('tile_size', self._tile_size)
