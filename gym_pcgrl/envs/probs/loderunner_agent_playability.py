import copy
import numpy as np
import time


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

# returns the reachable neighbours of a location(without the digging action)
def get_neighbours_no_dig(level, row, col, num_cols=32, num_rows=22):
    left_end = 0
    right_end = num_cols-1#15 #31
    top = 0
    bottom = num_rows-1#10 #21
    neighbours = []
    #if player is on the lowest row  
    if row==bottom:
        if col!=left_end and level[row][col-1]!=1 and level[row][col-1]!=4: neighbours.append((row,col-1))
        if col!=right_end and level[row][col+1]!=1 and level[row][col+1]!=4: neighbours.append((row,col+1))
        if level[row][col]==2 and level[row-1][col]!=1 and level[row-1][col]!=4: neighbours.append((row-1,col))

    #if current position is ladder or rope
    elif level[row][col] == 2 or level[row][col] == 3:
        #if player is not on the lowest row
        if row!=bottom:                    
            if level[row][col]==2 and row!=top and level[row-1][col]!=1 and level[row-1][col]!=4: neighbours.append((row-1,col))
            if level[row+1][col]!=1 and level[row+1][col]!=4: neighbours.append((row+1,col))
            if col!=left_end and (level[row][col-1]==2 or level[row][col-1]==3): neighbours.append((row,col-1))
            if col!=left_end and (level[row][col-1]==5 or level[row][col-1]==0) and (
                level[row+1][col-1]==1 or level[row+1][col-1]==4 or level[row+1][col-1]==2): neighbours.append((row,col-1))
            if col!=right_end and (level[row][col+1]==2 or level[row][col+1]==3): neighbours.append((row,col+1))  
            if col!=right_end and (level[row][col+1]==5 or level[row][col+1]==0) and (
                level[row+1][col+1]==1 or level[row+1][col+1]==4 or level[row+1][col+1]==2): neighbours.append((row,col+1))
            if col!=left_end and (level[row][col-1]==5 or level[row][col-1]==0) and (
                level[row+1][col-1]!=1 and level[row+1][col-1]!=4 and level[row+1][col-1]!=2): neighbours.append((row+1,col-1))
            if col!=right_end and (level[row][col+1]==5 or level[row][col+1]==0) and (
                level[row+1][col+1]!=1 and level[row+1][col+1]!=4 and level[row+1][col+1]!=2): neighbours.append((row+1,col+1))
    #if current position is empty or gold or enemy 
    elif level[row][col]==0 or level[row][col]==5:
        #if player is not on the lowest row
        if row!=bottom:
            #below is empty or rope or gold
            if level[row+1][col]!=1 and level[row+1][col]!=4:neighbours.append((row+1,col))
            #below is block or ladder
            if level[row+1][col] ==1 or level[row+1][col] ==4 or level[row+1][col] ==2:
                if col!=left_end and (level[row][col-1]==2 or level[row][col-1]==3): neighbours.append((row,col-1))
                if col!=left_end and (level[row][col-1]==5 or level[row][col-1]==0) and (
                    level[row+1][col-1]==1 or level[row+1][col-1]==4 or level[row+1][col-1]==2): neighbours.append((row,col-1))
                if col!=right_end and (level[row][col+1]==2 or level[row][col+1]==3): neighbours.append((row,col+1))  
                if col!=right_end and (level[row][col+1]==5 or level[row][col+1]==0) and (
                    level[row+1][col+1]==1 or level[row+1][col+1]==4 or level[row+1][col+1]==2): neighbours.append((row,col+1))
                if col!=left_end and (level[row][col-1]==5 or level[row][col-1]==0) and (
                    level[row+1][col-1]!=1 and level[row+1][col-1]!=4 and level[row+1][col-1]!=2): neighbours.append((row+1,col-1))
                if col!=right_end and (level[row][col+1]==5 or level[row][col+1]==0) and (
                    level[row+1][col+1]!=1 and level[row+1][col+1]!=4 and level[row+1][col+1]!=2): neighbours.append((row+1,col+1))
    return neighbours  


# returns the reachable neighbours of a location(the player can dig only one brick tile)
def get_neighbours_one_dig(level, row, col, num_cols=32, num_rows=22):
    left_end = 0
    right_end = num_cols - 1 #15#31
    top = 0
    bottom = num_rows - 1 #10#21
    neighbours = []
    #print(row,col)       
    #if player is on the lowest row  
    if row==bottom:
        if col!=left_end and level[row][col-1]!=1 and level[row][col-1]!=4: neighbours.append((row,col-1))
        if col!=right_end and level[row][col+1]!=1 and level[row][col+1]!=4: neighbours.append((row,col+1))
        if level[row][col]==2 and level[row-1][col]!=1 and level[row-1][col]!=4: neighbours.append((row-1,col))

    #if current position is ladder or rope
    elif level[row][col] == 2 or level[row][col] == 3 or level[row][col] == 1:
        #if player is not on the lowest row
        if row!=bottom:                    
            if level[row][col]==2 and row!=top and level[row-1][col]!=1 and level[row-1][col]!=4: neighbours.append((row-1,col))
            if level[row+1][col]!=1 and level[row+1][col]!=4: neighbours.append((row+1,col))
            if col!=left_end and (level[row][col-1]==2 or level[row][col-1]==3): neighbours.append((row,col-1))
            if col!=left_end and (level[row][col-1]==5 or level[row][col-1]==0) and (
                level[row+1][col-1]==1 or level[row+1][col-1]==4 or level[row+1][col-1]==2): neighbours.append((row,col-1))
            if col!=right_end and (level[row][col+1]==2 or level[row][col+1]==3): neighbours.append((row,col+1))  
            if col!=right_end and (level[row][col+1]==5 or level[row][col+1]==0) and (
                level[row+1][col+1]==1 or level[row+1][col+1]==4 or level[row+1][col+1]==2): neighbours.append((row,col+1))
            if col!=left_end and (level[row][col-1]==5 or level[row][col-1]==0) and (
                level[row+1][col-1]!=1 and level[row+1][col-1]!=4 and level[row+1][col-1]!=2): neighbours.append((row+1,col-1))
            if col!=right_end and (level[row][col+1]==5 or level[row][col+1]==0) and (
                level[row+1][col+1]!=1 and level[row+1][col+1]!=4 and level[row+1][col+1]!=2): neighbours.append((row+1,col+1))
            if col!=left_end and row!=bottom-1 and level[row+1,col-1]==1 and (
                level[row,col-1]==0 or level[row,col-1]==5):neighbours.append((row+1,col-1))
            if col!=right_end and row!=bottom-1 and level[row+1,col+1]==1 and (
                level[row,col+1]==0 or level[row,col+1]==5):neighbours.append((row+1,col+1))
    #if current position is empty or gold or enemy 
    elif level[row][col]==0 or level[row][col]==5:
        #if player is not on the lowest row
        if row!=bottom:
            #below is empty or rope or gold
            if level[row+1][col]!=1 and level[row+1][col]!=4:neighbours.append((row+1,col))
            #below is block or ladder
            if level[row+1][col] ==1 or level[row+1][col] ==4 or level[row+1][col] ==2:
                if col!=left_end and (level[row][col-1]==2 or level[row][col-1]==3): neighbours.append((row,col-1))
                if col!=left_end and (level[row][col-1]==5 or level[row][col-1]==0) and (
                    level[row+1][col-1]==1 or level[row+1][col-1]==4 or level[row+1][col-1]==2): neighbours.append((row,col-1))
                if col!=right_end and (level[row][col+1]==2 or level[row][col+1]==3): neighbours.append((row,col+1))  
                if col!=right_end and (level[row][col+1]==5 or level[row][col+1]==0) and (
                    level[row+1][col+1]==1 or level[row+1][col+1]==4 or level[row+1][col+1]==2): neighbours.append((row,col+1))
                if col!=left_end and (level[row][col-1]==5 or level[row][col-1]==0) and (
                    level[row+1][col-1]!=1 and level[row+1][col-1]!=4 and level[row+1][col-1]!=2): neighbours.append((row+1,col-1))
                if col!=right_end and (level[row][col+1]==5 or level[row][col+1]==0) and (
                    level[row+1][col+1]!=1 and level[row+1][col+1]!=4 and level[row+1][col+1]!=2): neighbours.append((row+1,col+1))
                if col!=left_end and level[row+1,col-1]==1 and row!=bottom-1 and(level[row,col-1]==0 
                    or level[row,col-1]==5):neighbours.append((row+1,col-1))
                if col!=right_end and level[row+1,col+1]==1 and row!=bottom-1 and(level[row,col+1]==0 
                    or level[row,col+1]==5):neighbours.append((row+1,col+1))
    return neighbours    



# returns the locations of the golds in the level
def get_gold_locs(level):
    golds = []
    for i in range(len(level)):
        for j in range(len(level[0])):
            if level[i][j]==5:
                golds.append((i,j))
    return golds

# returns the player location(starting point)
def get_starting_point(lvl):
    map = copy.copy(lvl)
    row=0
    col=0
    for i in range(len(map)):
        for j in range(len(map[0])):
            if map[i][j]==7:
                row=i
                col=j
                map[i][j]=0
            elif map[i][j]==6:
                map[i][j]=0
    return map,row,col

# returns all reachable locations in the level(from (x,y) location) using flood fill algorithm
def flood_fill(arr, x, y, num_cols=32, num_rows=22, digging=False):  
    # print(f"arr: {arr}, x: {x}, y: {y}")  
    if digging:
        find_neighbours = get_neighbours_one_dig
    else:
        find_neighbours = get_neighbours_no_dig
        
    to_visit = []
    visited = []
    to_visit.append((x,y))
    while len(to_visit) > 0:
        curr = to_visit.pop(0)
        neighbours = find_neighbours(arr, curr[0], curr[1], num_cols=num_cols, num_rows=num_rows)
        if len(neighbours) > 0:
            for n in neighbours:
                if n not in to_visit and n not in visited:
                    to_visit.append(n)
        visited.append(curr)     
    return visited


# returns reachable golds from each golds
def find_all_edges(level, digging, num_cols=32, num_rows=22):
    all_visited = []
    edges = []
    golds = get_gold_locs(level)
    map2d,row,col = get_starting_point(level)
    visited = flood_fill(map2d,row,col, num_cols=num_cols, num_rows=num_rows, digging=digging)
    all_visited.extend([(v,int_str_tiles_map[int(map2d[v[0]][v[1]])]) for v in visited])
    collected = [item for item in golds if item in visited and item!=(row,col)]
    for c in collected:
        edges.append(((row,col),c))

    for g in golds:
        visited = flood_fill(map2d,g[0],g[1],num_cols=num_cols, num_rows=num_rows, digging=digging)
        all_visited.extend([(v,int_str_tiles_map[int(map2d[v[0]][v[1]])]) for v in visited])
        collected = [item for item in golds if item in visited and item != g]
        for c in collected:
            edges.append((g,c))
    return (row, col), golds, edges, all_visited



def is_connected(start, goal, edges):
    #print("check",start, goal)
    if (start, goal) in edges:
        return True
    else:
        return False

# starting from player location, finds the best sequence of reachable golds
def find_golds_seq(start, golds, edges, clock, max_time):
    if len(golds) == 0:
        return []
    bestSeq = []
    for g in golds:
        if len(bestSeq) == len(golds) or time.time()-clock > max_time:
            break
        connected = is_connected(start, g, edges)
        if connected:
            rem_golds = golds.copy()
            rem_golds.remove(g)
            sequence = find_golds_seq(g, rem_golds, edges, clock, max_time)
            if bestSeq is None or len(sequence) + 1 > len(bestSeq):
                bestSeq = [g] + sequence
    return bestSeq

'''
returns the sequence of reachable golds from the player location

level: 2D int array---the level, the level must have a player
max_time: int--- maximum time given for the search
digging: bool--- the player can dig or not 
'''
def chk_playability(level, max_time=3, num_cols=32, num_rows=22, digging=True):
    root, golds, edges, all_visited = find_all_edges(level, digging, num_cols=num_cols, num_rows=num_rows)
    clock = time.time()
    seq = find_golds_seq(root, golds, edges, clock, max_time)
    return seq, root, golds, edges, list(set(all_visited))
    
    
    

