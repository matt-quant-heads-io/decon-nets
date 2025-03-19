"""
Run a trained agent and get generated maps
"""
import os

import torch
from tqdm import tqdm
import numpy as np


DOMAIN_TO_INT_TILE_MAP = {
    "empty": 0,
    "solid": 1,
    "player": 2,
    "key": 3,
    "door": 4,
    "bat": 5,
    "scorpion": 6,
    "spider": 7
}
DOMAIN_TO_CHAR_TO_STR_TILE_MAP = {
    "g": "door",
    "+": "key",
    "A": "player",
    "1": "bat",
    "2": "spider",
    "3": "scorpion",
    "w": "solid",
    ".": "empty"
}

REV_TILES_MAP = {v:k for k,v in DOMAIN_TO_CHAR_TO_STR_TILE_MAP.items()}

def str_arr_from_int_arr(map, str_to_int_map):
    translation_map = {v: k for k, v in str_to_int_map.items()}
    str_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(translation_map[map[row_idx][col_idx]])
        str_map.append(new_row)

    return str_map


def str_map_to_char_map(str_map):
    level_str = ""
    for row in str_map:
        for col in row:
            level_str += REV_TILES_MAP[col]

    return level_str


def convert_int_map_to_str_map(state, int_to_str_tiles_map):
    str_map = []
    for row_i, row in enumerate(state):
        for col_i, col in enumerate(row):
            str_map.append(int_to_str_tiles_map[state[row_i][col_i]])
    return np.array(str_map).reshape(state.shape)

def convert_str_map_to_int_map(state, str_to_int_tiles_map):
    int_map = []
    for row_i, row in enumerate(state):
        for col_i, col in enumerate(row):
            int_map.append(str_to_int_tiles_map[col])

    return np.array(int_map).reshape(*state.shape)


def convert_onehot_map_to_int_map(state):
    int_map = []
    for row_i, row in enumerate(state):
        for col_i, col in enumerate(row):
            idx = np.where(col == 1)
            int_map.append(idx)
    return np.array(int_map).reshape(*state.shape[:2])


def transform(obs, x, y, obs_size):
    map = obs
    size = obs_size[0]
    pad = obs_size[0] // 2
    padded = np.pad(map, pad, constant_values=1)
    cropped = padded[y: y + size, x: x + size]
    
    return cropped

def convert_from_env_obs_to_model_obs(obs, x, y, obs_size, int_to_str_tiles_map, prob_obj):
    obs = transform(obs, x, y, obs_size)
    str_map = convert_int_map_to_str_map(obs, int_to_str_tiles_map)
    img = prob_obj.render(str_map)
    img = img.convert('RGB')

    return img


def convert_char_maps_to_int_maps(char_maps_list):
    all_levels = []
    for char_map in char_maps_list:
        all_levels.append(np.array([DOMAIN_TO_INT_TILE_MAP[DOMAIN_TO_CHAR_TO_STR_TILE_MAP[char]] for char in char_map]))

    return np.array(all_levels)

def get_int_maps_from_char_maps_paths(list_of_paths_to_char_maps):
    def to_2d_array_level(file_name, tiles_map):
        level = []

        with open(file_name, "r") as f:
            rows = f.readlines()
            for row in rows:
                new_row = []
                for char in row:
                    if char != "\n":
                        new_row.append(tiles_map[char])
                level.append(new_row)

        return level
    
    def int_arr_from_str_arr(map, int_arr_from_str_arr):
        int_map = []
        for row_idx in range(len(map)):
            new_row = []
            for col_idx in range(len(map[0])):
                new_row.append(int_arr_from_str_arr[map[row_idx][col_idx]])
            int_map.append(new_row)
        return int_map 

    return [int_arr_from_str_arr(to_2d_array_level(p, DOMAIN_TO_CHAR_TO_STR_TILE_MAP), DOMAIN_TO_INT_TILE_MAP) for p in list_of_paths_to_char_maps]


def calc_diversity(test_lvls, goal_lvls, cuttoff):
    def hamm_dist(lvl1, lvl2):
        value = np.abs(lvl1.flatten() - lvl2.flatten())
        value[value > 0] = 1
        return value.sum()
        
    def direction_diversity(lvls, goals, distfn=hamm_dist):
        dists = []
        lvls = np.array(lvls)
        goals = np.array(goals)
        for lvl in lvls:
            min_dist = -1
            for goal in goals:
                new_dist = distfn(lvl, goal)
                if min_dist < 0 or new_dist < min_dist:
                    min_dist = new_dist
            dists.append(min_dist)
        return np.array(dists)

    def get_subset_diversity(lvls, goal, cuttoff, distfn=hamm_dist):
        dists = direction_diversity(lvls, goal, distfn)
        return lvls[dists >= cuttoff]

    def greedy_set_diversity(lvls, cuttoff, distfn=hamm_dist):
        indeces = set()
        for i,lvl in enumerate(tqdm(lvls, leave=False)):
            if i in indeces:
                continue
            values = direction_diversity(lvls, [lvl], distfn)
            temp = np.where(values < cuttoff)[0]
            if len(temp) > 1:
                temp = temp[temp > i]
                indeces.update(temp)
        return np.delete(lvls, list(indeces), axis=0)


    def ogreedy_set_diversity(lvls, cuttoff, distfn=hamm_dist):
        extra_info = []
        repeat = np.zeros(len(lvls))
        for i,lvl in enumerate(tqdm(lvls, leave=False, desc="Sorting By Repeatition")):
            values = direction_diversity(lvls, [lvl], distfn)
            values[i] = 10000
            repeat[i] = (values < cuttoff).sum()
            
            temp_info = {}
            temp_info['cuttoff'] = (values < cuttoff).sum()
            temp_info['worse'] = values.min()
            temp_info['identical'] = (values == 0).sum()
            temp_info['worse_identical'] = (values == temp_info['worse']).sum()
            extra_info.append(temp_info)
        return greedy_set_diversity(lvls[repeat.argsort()], cuttoff, distfn), extra_info

    unique_goal_lvls = get_subset_diversity(test_lvls, goal_lvls, cuttoff)
    div_wrt_goal = len(unique_goal_lvls) / len(test_lvls)

    unique_ogreedy_lvls, extra = ogreedy_set_diversity(test_lvls, cuttoff)
    div_wrt_internal = len(unique_ogreedy_lvls) / len(test_lvls)

    mean_div = (div_wrt_goal + div_wrt_internal) / 2
    return mean_div


def infer(network, env, **kwargs):
    """
    Run inference using a trained network on the given environment.
    
    Args:
        network: The trained neural network model
        env: The environment to run inference on
        **kwargs: Additional arguments including:
            - trials: Number of trials to run
            - verbose: Whether to print progress
    
    Returns:
        tuple: (solve_percentage, diversity_score)
    """
    solved = 0
    unsolved = 0
    rewards = []
    solved_maps = []
    entropies_per_trial = []
    
    trials = kwargs.get('trials', 1)
    verbose = kwargs.get('verbose', False)
    cuttoff = kwargs.get('div_cutoff', 0.10)
    goal_map_path = kwargs.get('goal_map_path', None)
    
    obs_pos_dict, info = env.unwrapped.reset() 
    y, x = obs_pos_dict['pos'] 
    y, x = y.item(), x.item()
    obs = obs_pos_dict['map'] 
    
    dones = False
    network.network.eval()


    for i in range(trials):
        entropy_per_trial = []
        while not dones:
            output, entropy = network.policy.predict(torch.from_numpy(obs), x, y)
            action = output.item()

            entropy_per_trial.append(entropy)
            
            next_obs_pos_dict, reward, dones, truncated, info = env.step(action)
            y, x = next_obs_pos_dict['pos']
            x, y, = x.item(), y.item()   
            
            if dones:
                if info["solved"]:
                    solved += 1
                    solved_pct = round(float(solved) / (float(solved) + float(unsolved)), 2) * 100
                    reward = info["reward"]
                    rewards.append(reward)
                    
                    if verbose:
                        print(f"Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}")
                        print(f"Mean reward: {sum(rewards)/len(rewards)}")
                    
                    char_map = str_map_to_char_map(info["final_map"])
                    solved_maps.append(char_map)
                else:
                    unsolved += 1
                    if verbose:
                        solved_pct = round(float(solved) / (float(solved) + float(unsolved)), 2) * 100
                        print(f"Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}")
                break
                
        dones = False
        obs_pos_dict, info = env.unwrapped.reset() 
        y, x = obs_pos_dict['pos'] 
        y, x = y.item(), x.item()
        obs = obs_pos_dict['map'] 
        entropies_per_trial.append(sum(entropy_per_trial)/len(entropy_per_trial))
    
    # Calculate final metrics
    goal_maps_path = "./goal_maps/zelda"
    list_of_paths_to_char_maps = [f"{goal_maps_path}/{gmap}" for gmap in os.listdir(goal_maps_path) if gmap.endswith(".txt")]
    goal_maps_as_int = get_int_maps_from_char_maps_paths(list_of_paths_to_char_maps)
    solve_percentage = round(float(solved) / float(trials) * 100, 2)
    mean_entropy = sum(entropies_per_trial) / len(entropies_per_trial)

    if len(solved_maps) == 0:
        return solve_percentage, 0.0, mean_entropy
 
    solved_maps_as_int = convert_char_maps_to_int_maps(solved_maps)

    diversity_score = calc_diversity(np.array(goal_maps_as_int), np.array(solved_maps_as_int), cuttoff) 
    
    
    return solve_percentage, diversity_score, mean_entropy



