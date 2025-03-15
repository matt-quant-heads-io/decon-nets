import os
import math
import copy
from collections import OrderedDict
import random
from timeit import default_timer as timer
from datetime import timedelta
import argparse
import datetime

import numpy as np
import pandas as pd
from gym import error
import json
import tqdm


from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.loderunner_prob import LRProblem
from gym_pcgrl.wrappers import CroppedImagePCGRLWrapper
from gym_pcgrl.envs.helper import (
    get_tile_locations,
    calc_num_regions,
    calc_certain_tile,
    run_dikjstra,
    get_string_map,
)

import utils
from gym_pcgrl.envs.probs.loderunner_agent_playability import chk_playability


DOMAIN_TO_PROB_OBJ_MAP = {
    "loderunner": LRProblem,
    "zelda": ZeldaProblem
}
DOMAIN_TO_INT_TILE_MAP = {
    "zelda": {
        "empty": 0,
        "solid": 1,
        "player": 2,
        "key": 3,
        "door": 4,
        "bat": 5,
        "scorpion": 6,
        "spider": 7,
    },
    "loderunner": {
        "empty": 0,
        "brick": 1, 
        "ladder": 2, 
        "rope": 3, 
        "solid": 4, 
        "gold": 5, 
        "enemy": 6, 
        "player": 7
    }
}
DOMAIN_TO_CHAR_TO_STR_TILE_MAP = {
    "zelda":{
        "g": "door",
        "+": "key",
        "A": "player",
        "1": "bat",
        "2": "spider",
        "3": "scorpion",
        "w": "solid",
        ".": "empty",
    },
    "loderunner": {
        ".":"empty",
        "b":"brick", 
        "#":"ladder", 
        "-":"rope", 
        "B":"solid", 
        "G":"gold", 
        "E":"enemy", 
        "M":"player"
    }
}
os.system("source ../set_project_root.sh")
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")


def gen_random_map(random, width, height, prob):
    map = random.choice(
        list(prob.keys()), size=(height, width), p=list(prob.values())
    ).astype(np.uint8)
    return map


def compute_hamm_dist(random_map, goal):
    hamming_distance = 0.0
    for i in range(len(random_map)):
        for j in range(len(random_map[0])):
            if random_map[i][j] != goal[i][j]:
                hamming_distance += 1
    return float(hamming_distance / (len(random_map) * len(random_map[0])))


def find_closest_goal_map(random_map, goal_set_idxs, goal_maps_filepath, char_to_string_tiles_map, str_to_int_tiles_map):
    smallest_hamming_dist = math.inf
    filepath = goal_maps_filepath
    goal_maps = [f for f in os.listdir(goal_maps_filepath) if '.txt' in f]
    closest_map = curr_goal_map = int_arr_from_str_arr(
        to_2d_array_level(f"{goal_maps_filepath}/{goal_maps[0]}", char_to_string_tiles_map), str_to_int_tiles_map
    )

    for next_curr_goal_map_fp in goal_maps:
        next_curr_goal_map = int_arr_from_str_arr(
            to_2d_array_level(f"{goal_maps_filepath}/{next_curr_goal_map_fp}", char_to_string_tiles_map), str_to_int_tiles_map
        )
        temp_hamm_distance = compute_hamm_dist(random_map, next_curr_goal_map)
        if temp_hamm_distance < smallest_hamming_dist:
            closest_map = next_curr_goal_map
            smallest_hamming_dist = temp_hamm_distance

    return closest_map


def gen_pod_transitions(random_map, goal_map, traj_len, obs_size, controllable=False, randomize_sequence=True, prob_obj=None, str_to_int_map=None, xys=None):
    string_map_for_map_stats = str_arr_from_int_arr(goal_map, str_to_int_map)

    # Targets
    if controllable:
        new_map_stats_dict = prob_obj.get_stats(string_map_for_map_stats)
        num_enemies = new_map_stats_dict["enemies"]
        nearest_enemy = new_map_stats_dict["nearest-enemy"]
        path_length = new_map_stats_dict["path-length"]
        conditional_diffs = []
        

    if xys is None:
        xys = []
        for row_i, row in enumerate(random_map):
            for col_i, col in enumerate(row):
                xys.append((row_i, col_i))
            
    # import pdb; pdb.set_trace()
    # xys = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10)]
    
    if randomize_sequence:
        random.shuffle(xys)

    steps = 0
    obs = copy.copy(random_map)
    cumulative_reward = 0.0
    returns = []
    states = []
    actions = []
    dones = []
    next_states = []
    is_start = True
    rewards = []
    is_starts = []
    while steps < traj_len and len(xys) > 0:
        steps += 1
        y, x = xys.pop()
        action = goal_map[y][x]
        prev_obs = copy.copy(obs)
        row = obs[y]
        row[x] = action 
        obs[y] = row
        # print(f"stats1:{}")
        # print(f"stats2: {}")
        # current_reward = prob_obj.get_reward(prob_obj.get_stats(str_arr_from_int_arr(transform(obs, x, y, obs_size),str_to_int_map)), prob_obj.get_stats(str_arr_from_int_arr(transform(prev_obs, x, y, obs_size),str_to_int_map)))
        # cumulative_reward += current_reward
        # is_done = prob_obj.get_episode_over(prob_obj.get_stats(str_arr_from_int_arr(transform(obs, x, y, obs_size),str_to_int_map)), prob_obj.get_stats(str_arr_from_int_arr(transform(prev_obs, x, y, obs_size),str_to_int_map)))
        str_int_obs_curr = str_arr_from_int_arr(obs, str_to_int_map)
        str_int_obs_prev = str_arr_from_int_arr(prev_obs, str_to_int_map)
        curr_obs_stats = prob_obj.get_stats(str_int_obs_curr)
        prev_obs_stats = prob_obj.get_stats(str_int_obs_prev)
        str_onehot_obs_curr = str_arr_from_int_arr(transform(obs, x, y, obs_size), str_to_int_map)
        
        str_onehot_obs_prev = str_arr_from_int_arr(transform(prev_obs, x, y, obs_size),str_to_int_map)
        current_reward = prob_obj.get_reward(curr_obs_stats, prev_obs_stats)
        cumulative_reward += current_reward
        is_done = prob_obj.get_episode_over(curr_obs_stats, prev_obs_stats)
        dones.append(is_done)
        rewards.append(current_reward)
        if is_start:
            is_starts.append(is_start)
            is_start = False
        else:
            is_starts.append(is_start)
        
        states.append(transform(prev_obs, x, y, obs_size))
        next_states.append(transform(obs, x, y, obs_size))
        actions.append(action)


        if controllable:
            string_map_for_map_stats = str_arr_from_int_arr(obs)
            new_map_stats_dict = prob_obj.get_stats(string_map_for_map_stats)
            enemies_diff = num_enemies - new_map_stats_dict["enemies"]
            if enemies_diff > 0:
                enemies_diff = 3
            elif enemies_diff < 0:
                enemies_diff = 1
            else:
                enemies_diff = 2
    
            nearest_enemies_diff = nearest_enemy - new_map_stats_dict["nearest-enemy"]
            if nearest_enemies_diff > 0:
                nearest_enemies_diff = 3
            elif nearest_enemies_diff < 0:
                nearest_enemies_diff = 1
            else:
                nearest_enemies_diff = 2
    
            path_diff = path_length - new_map_stats_dict["path-length"]
            if path_diff > 0:
                path_diff = 3
            elif path_diff < 0:
                path_diff = 1
            else:
                path_diff = 2
    
            conditional_diffs.append(
                (enemies_diff, nearest_enemies_diff, path_diff)
            )     

    returns = [cumulative_reward/float(steps) for _ in range(len(states))]
     
    if controllable:
        return next_states, states, actions, returns, conditional_diffs, dones, is_starts, rewards
    else:
        return next_states, states, actions, returns, dones, is_starts, rewards


def transform(obs, x, y, obs_size):
    map = obs
    size = obs_size[0]
    pad = obs_size[0] // 2
    padded = np.pad(map, pad, constant_values=1)
    cropped = padded[y: y + size, x: x + size]
    
    return cropped
    

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


def str_arr_from_int_arr(map, str_to_int_map):
    translation_map = {v: k for k, v in str_to_int_map.items()}
    str_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(translation_map[map[row_idx][col_idx]])
        str_map.append(new_row)

    return str_map


def convert_state_to_onehot(state, obs_dim, act_dim):
    new_state = []
    new_state_oh = []
    for row in range(len(state)):
        new_row = []
        for col in range(len(state[0])):
            cell_oh = [0]*act_dim
            cell_oh[int(state[row][col])]=1
            # new_row.append(cell_oh)
            # print(cell_oh)
            new_state_oh.append(cell_oh)
            
            # new_state.append(constants.DOMAIN_VARS_ZELDA["int_map"][state[row][col]])
            
    # import pdb; pdb.set_trace()
    return np.array(new_state_oh).reshape((obs_dim[0], obs_dim[1], act_dim))


def int_from_oh(state, obs_dim):
    int_map = []
    for row in range(len(state)):
        new_row = []
        for col in range(len(state[0])):
            cell = np.argmax(state[row][col])
            new_row.append(cell)

        int_map.append(new_row)
    return np.array(int_map).reshape(obs_dim[0], obs_dim[1])


def gen_trajectories(domain, num_episodes, mode_of_output, action_dim, obs_dim, traj_len, sampling_strat):
    goal_maps_root_dir = f"{PROJECT_ROOT}/goal_maps/{domain}"
    goal_maps_set = [i for i in range(0, len(sorted(os.listdir(goal_maps_root_dir)[:5])))]
    prob_obj = DOMAIN_TO_PROB_OBJ_MAP[domain]()
    # random.shuffle(goal_maps_set)
    goal_set_idxs = goal_maps_set
    rng, _ = utils.np_random(None)
    char_to_string_tiles_map = DOMAIN_TO_CHAR_TO_STR_TILE_MAP[domain]
    str_to_int_tiles_map = DOMAIN_TO_INT_TILE_MAP[domain]
    start_maps = [
        gen_random_map(
            rng,
            prob_obj._width,
            prob_obj._height,
            {str_to_int_tiles_map[s]: p for s,p in prob_obj._prob.items()},
        )
        for _ in range(num_episodes)
    ]
    goal_maps = [
        find_closest_goal_map(start_map, goal_set_idxs, goal_maps_root_dir, char_to_string_tiles_map, str_to_int_tiles_map) for start_map in start_maps
    ]
    
    action_dim = (action_dim,)
    obs_dim = (obs_dim, obs_dim, action_dim[0])
    reward_approximator_trajectory = []
    xys = None
    for epsiode_num, (start_map, goal_map) in tqdm.tqdm(enumerate(zip(start_maps, goal_maps))):
        if sampling_strat == "player_agent_trace_start_map":
            xys = []
            for trace_tuple in chk_playability(start_map)[-1]:
                xy, tile_type = trace_tuple
                xys.append((xy[0], xy[1]))
                
        elif sampling_strat == "player_agent_trace_goal_map":
            xys = []
            # print(type(goal_map))
            for trace_tuple in chk_playability(np.array(goal_map))[-1]:
                xy, tile_type = trace_tuple
                xys.append((xy[0], xy[1]))

        # import pdb; pdb.set_trace()
        next_states, states, actions, returns, dones, is_starts, rewards = gen_pod_transitions(start_map, goal_map, traj_len, obs_dim, prob_obj=prob_obj, str_to_int_map=str_to_int_tiles_map, xys=xys)
        for next_state, state, action, ret, done, is_start, reward in zip(next_states, states, actions, returns, dones, is_starts, rewards):
            reward_approximator_trajectory.append({
                "next_state": next_state,
                "state": state,
                "action": action,
                "return": ret,
                "done": done,
                "is_start": is_start,
                "rewards": reward
            })

        print(f"generated {epsiode_num+1} episodes")
    
    tuple_trajectories = []
    for json_dict in reward_approximator_trajectory:
        ret = json_dict["return"]
        done = json_dict["done"]
        if mode_of_output == "onehot":
            action_oh = [0]*action_dim[0]
            action_oh[json_dict["action"]] = 1
            tuple_trajectories.append((convert_state_to_onehot(json_dict["next_state"], obs_dim, action_dim[0]), convert_state_to_onehot(json_dict["state"], obs_dim, action_dim[0]), np.array(action_oh), np.array([ret]), np.array([done]), np.array([json_dict["is_start"]]), np.array([json_dict["rewards"]])))
        else:
            tuple_trajectories.append((json_dict["next_state"], json_dict["state"], json_dict["action"], np.array([ret]), np.array([done]), np.array([json_dict["is_start"]]), np.array([json_dict["rewards"]])))
    
    # random.shuffle(tuple_trajectories)
    batch_size = 100000 #num_episodes//

    if mode_of_output == "onehot":
        # import pdb; pdb.set_trace()
        expert_observations = np.empty((batch_size,) + obs_dim)
        expert_next_observations = np.empty((batch_size,) + obs_dim)
        expert_actions = np.empty((batch_size,) + action_dim)
    else:
        expert_observations = np.empty((batch_size,) + obs_dim)
        expert_next_observations = np.empty((batch_size,) + obs_dim)
        expert_actions = np.empty((batch_size,) + (1,))    

    expert_returns = np.empty((batch_size,) + (1,))
    expert_dones = np.empty((batch_size,) + (1,))
    expert_is_starts = np.empty((batch_size,) + (1,))
    expert_rewards = np.empty((batch_size,) + (1,))

    reward_approximator_trajectory.append({
        "next_state": next_state,
        "state": state,
        "action": action,
        "rewards": ret,
        "done": done,
        "is_start": is_start,
        "rewards": rewards
    })

    
    from itertools import islice

    def batch_data(iterable, n):
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

    saves = 4  
    internal_idx = 0
    for i, (next_obs, obs, act, returns, done, is_start, reward) in enumerate(tuple_trajectories):
        # import pdb; pdb.set_trace()
        print(f"i: {i}")
        expert_next_observations[internal_idx] = next_obs
        expert_observations[internal_idx] = obs
        expert_actions[internal_idx] = act
        expert_rewards[internal_idx] = reward
        expert_dones[internal_idx] = done
        expert_is_starts[internal_idx] = is_start
        expert_rewards[internal_idx] = reward

        if (i + 1) % batch_size == 0:
            numpy_archive_filename = f"{PROJECT_ROOT}/data/{domain}/pod_trajs_{sampling_strat}_{saves+1}_sampling.npz"
            np.savez_compressed(
                  numpy_archive_filename,
                  actions=expert_actions,
                  episode_returns=expert_returns,
                  rewards=expert_rewards,
                  obs=expert_observations,
                  episode_starts=expert_is_starts
              )
            if mode_of_output == "onehot":
                # import pdb; pdb.set_trace()
                expert_observations = np.empty((batch_size,) + obs_dim)
                expert_next_observations = np.empty((batch_size,) + obs_dim)
                expert_actions = np.empty((batch_size,) + action_dim)
            else:
                expert_observations = np.empty((batch_size,) + obs_dim)
                expert_next_observations = np.empty((batch_size,) + obs_dim)
                expert_actions = np.empty((batch_size,) + (1,))   

            expert_returns = np.empty((batch_size,) + (1,))
            expert_dones = np.empty((batch_size,) + (1,))
            expert_is_starts = np.empty((batch_size,) + (1,))
            expert_rewards = np.empty((batch_size,) + (1,))
            saves += 1
            internal_idx = 0
        else:
            internal_idx+=1

    print(f"Saved file as {numpy_archive_filename}")
    
# PROJECT_ROOT=/home/jupyter-msiper/bootstrapping-pcgrl python gen_pod_traj.py --domain loderunner --num_episodes 2 --mode_of_output onehot --action_dim 8 --obs_dim 64 --traj_len 704 --sampling_strat normal
def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--domain", help="domain with which to run PoD", type=str, default="loderunner", choices=["lego", "loderunner", "zelda"])
    argparser.add_argument("--sampling_strat", type=str, default="normal", choices=["normal", "player_agent_trace_start_map", "player_agent_trace_goal_map"])
    argparser.add_argument("-n", "--num_episodes", help="number of episodes to run PoD", type=int, default=10_000)
    argparser.add_argument("-m", "--mode_of_output", help="Output mode of the PoD algorithm", type=str, default="onehot", choices=["onehot", "integer", "image", "string"])
    argparser.add_argument("-a", "--action_dim", help="Action dimension", type=int)
    argparser.add_argument("-o", "--obs_dim", help="Observation dimension", type=int)
    argparser.add_argument("-t", "--traj_len", help="Trajectory length", type=int)
    
    
    return argparser.parse_args()

    

def gen_pod_trajectories(args):
    gen_trajectories(args.domain, args.num_episodes, args.mode_of_output, args.action_dim, args.obs_dim, args.traj_len, args.sampling_strat)


def main():
    args = get_args()
    gen_trajectories(args.domain, args.num_episodes, args.mode_of_output, args.action_dim, args.obs_dim, args.traj_len, args.sampling_strat)
    #gen_pod_trajectories(args)
    
    
# NOTE: to run this --> PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python gen_expert_traj.py

main()
    
        