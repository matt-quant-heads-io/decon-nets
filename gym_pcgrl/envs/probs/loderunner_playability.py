from typing import List, Set, Tuple
from collections import deque



str_int_tiles_map = {
    "empty": 0,
    "brick": 1, 
    "ladder": 2, 
    "rope": 3, 
    "solid": 4, 
    "gold": 5, 
    "enemy": 6, 
    "player": 7
}
int_str_tiles_map = {v:k for k,v in str_int_tiles_map.items()}

def is_level_playable(level: List[List[str]]) -> bool:
    """
    Check if a Lode Runner level is playable.
    A level is playable if:
    1. There is exactly one player
    2. All gold pieces are reachable by the player
    
    Args:
        level: 2D list representing the level where:
            7 - Player
            5 - Gold
            3 - Rope
            2 - Ladder
            4 - Platform/Ground
            0 - Empty space
    
    Returns:
        bool: True if the level is playable, False otherwise
    """
    
    # Find player position and count gold pieces
    player_pos = None
    gold_positions = set()
    height = len(level)
    width = len(level[0])
    
    for y in range(height):
        for x in range(width):
            if level[y][x] == 7:
                if player_pos is not None:
                    return False  # More than one player
                player_pos = (x, y)
            elif level[y][x] == 5:
                gold_positions.add((x, y))
    
    if player_pos is None:
        return False  # No player found
    
    # BFS to find reachable positions
    reachable = set()
    queue = deque([player_pos])
    visited = {player_pos}
    
    def can_move_to(x: int, y: int) -> bool:
        """Check if player can move to the given position"""
        if not (0 <= x < width and 0 <= y < height):
            return False
        return True
    
    def has_platform_below(x: int, y: int) -> bool:
        """Check if there's a platform or ladder below the position"""
        if y + 1 >= height:
            return True  # Bottom of level counts as platform
        return level[y + 1][x] in [0, 2]
    
    while queue:
        x, y = queue.popleft()
        reachable.add((x, y))
        
        # Check all possible moves from current position
        current_tile = level[y][x]
        
        # Standing on platform or ladder
        if has_platform_below(x, y):
            # Move left/right
            for dx in [-1, 1]:
                new_x = x + dx
                if can_move_to(new_x, y) and (new_x, y) not in visited:
                    if has_platform_below(new_x, y):
                        queue.append((new_x, y))
                        visited.add((new_x, y))
            
            # Climb up ladder
            if current_tile == 2 and can_move_to(x, y - 1):
                if (x, y - 1) not in visited:
                    queue.append((x, y - 1))
                    visited.add((x, y - 1))
        
        # On a rope
        if current_tile == 3:
            # Move along rope
            for dx in [-1, 1]:
                new_x = x + dx
                if can_move_to(new_x, y) and level[y][new_x] == 3 and (new_x, y) not in visited:
                    queue.append((new_x, y))
                    visited.add((new_x, y))
        
        # Check for fall paths
        if not has_platform_below(x, y):
            fall_y = y + 1
            while fall_y < height and not has_platform_below(x, fall_y):
                if (x, fall_y) not in visited:
                    queue.append((x, fall_y))
                    visited.add((x, fall_y))
                fall_y += 1
            
            # Add landing position
            if fall_y < height and (x, fall_y) not in visited:
                queue.append((x, fall_y))
                visited.add((x, fall_y))
    
    # Check if all gold pieces are reachable
    return all(gold_pos in reachable for gold_pos in gold_positions)

# Example usage:
def test_level():
    level = [
        [4, 4, 4, 4, 4],
        [0, 5, 3, 5, 0],
        [4, 2, 4, 2, 4],
        [0, 7, 0, 0, 0],
        [4, 4, 4, 4, 4]
    ]
    
    result = is_level_playable(level)
    print(f"Level is playable: {result}")
    
    # Test level with unreachable gold
    level2 = [
        [4, 4, 4, 4, 4],
        [0, 5, 0, 5, 0],
        [4, 2, 4, 4, 4],
        [0, 7, 0, 0, 0],
        [4, 4, 4, 4, 4]
    ]
    
    result2 = is_level_playable(level2)
    print(f"Level 2 is playable: {result2}")

# if __name__ == "__main__":
#     test_level()