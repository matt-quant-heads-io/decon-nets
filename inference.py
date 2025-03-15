"""
Run a trained agent and get generated maps
"""
import json
import model

from utils import make_env, make_vec_envs
import os
import argparse
import torch as th
import pathlib
# from sl_model import CNN
# from gen_pod_traj import transform

from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.probs.loderunner_prob import LRProblem
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from model import CustomActorCriticPolicy, CustomCNNFeatureExtractor, WrappedNetwork
from model import CustomPPO
import numpy as np

import torch
from tqdm import tqdm
from einops import rearrange
import torch.nn as nn
import json
import torch as th
import numpy as np

# import utils

import torch.nn.functional as F
from utils import make_vec_envs as mkvenv



PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")

DOMAIN_TO_INT_TILE_MAP = {
    "empty": 0,
    "brick": 1, 
    "ladder": 2, 
    "rope": 3, 
    "solid": 4, 
    "gold": 5, 
    "enemy": 6, 
    "player": 7
}
DOMAIN_TO_CHAR_TO_STR_TILE_MAP =  {
    ".":"empty",
    "b":"brick", 
    "#":"ladder", 
    "-":"rope", 
    "B":"solid", 
    "G":"gold", 
    "E":"enemy", 
    "M":"player"
}

REV_TILES_MAP = {
    "door": "g",
    "key": "+",
    "player": "A",
    "bat": "1",
    "spider": "2",
    "scorpion": "3",
    "solid": "w",
    "empty": ".",
}    

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

    # import pdb; pdb.set_trace()
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



def infer(game, representation, experiment, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        # model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        # model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        # model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    elif game == "loderunner":
        kwargs['cropped_size'] = 64

    kwargs['render'] = True

    # env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    # import pdb; pdb.set_trace()
    # device = "cpu"

    # env = PcgrlEnv(prob="loderunner", rep="narrow")
    # prob_obj = LRProblem()

    # data = np.load("/home/jupyter-msiper/bootstrapping-pcgrl/data/loderunner/pod_trajs_2025-01-11 19:56:45.897043.npz")
    # start_state = data["obs"].astype(np.float32)[0]
    # int_state = convert_onehot_map_to_int_map(start_state)
    int_to_str_map = {v:k for k,v in DOMAIN_TO_INT_TILE_MAP.items()}
    # str_map = convert_int_map_to_str_map(int_state, int_to_str_map)
    
    # print(f"obs: {obs.shape}")
    # obs = env.reset()['map']
    # dt_obs = convert_from_env_obs_to_model_obs(env._rep._map, env._rep._x, env._rep._y, (64, 64, 8), int_to_str_map, prob_obj)
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': False,
        "change_percentage": 3.0,
        "trials": 1000,
        "verbose": True,
        "experiment": "supervised_training"
    }
    prob_obj = ZeldaProblem()
    env = mkvenv("zelda-narrow-v0", "narrow", None, 1, **kwargs)
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    # model = CustomPPO(CustomActorCriticPolicy, env=env, policy_kwargs=policy_kwargs, verbose=2, exp_path=f"./experiments/{game}/supervised_training", device = "cuda" if torch.cuda.is_available() else "cpu")
    model = CustomPPO(CustomActorCriticPolicy, env=env, policy_kwargs=policy_kwargs, verbose=2, exp_path=f"./experiments/{game}/supervised_training", device = "cpu")
    model.load_supervised_weights(
        f"/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/feature_extractor.pth",
        f"/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/mlp_extractor.pth",
        f"/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/action_net.pth",
        f"/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/value_net.pth",

    )
    model.policy.eval()
    # model.save("/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/model.zip")
    # model = CustomPPO.load("/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/model.zip")
    # obs_space = env.__dict__['envs'][0].env.observation_space
    # action_space = env.__dict__['envs'][0].env.action_space
    # slmodel = WrappedNetwork(obs_space, action_space, 1, features_dim=256, last_layer_dim_pi=256, last_layer_dim_vf=256)
    
    # #
    # slmodel.load_state_dict(torch.load("/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/sl_policy.pth", weights_only=True, map_location=torch.device('cpu')))
    # slmodel.eval()
    # import pdb; pdb.set_trace()
    # str_arr_from_int_arr()
    # import pdb; pdb.set_trace()
    obs = env.reset()
    # import pdb; pdb.set_trace()

    dones = False
    solved = 0
    unsolved = 0
    rewards = []
    results = {"solved_maps": [], 'num_boostrap_episodes': [], 'train_process': [], 'num_ppo_timesteps': [], 'num_boostrap_epochs': [], 'bootstrap_total_time': [], 'ppo_total_time': [], 'total_train_time': []}
    for i in range(kwargs.get('trials', 1)):
        while not dones:
            # obs = convert_from_env_obs_to_model_obs(obs, env.__dict__['envs'][0].env.pcgrl_env.env.env._rep._x, env.__dict__['envs'][0].env.pcgrl_env.env.env._rep._y, (22, 22, 8), int_to_str_map, prob_obj)
                
            # import pdb; pdb.set_trace()
            # output = model.predict(torch.argmax(torch.from_numpy(obs),dim=3).reshape(22,22))
            output, _ = model.policy.predict(torch.from_numpy(obs))

            action = output.item()
            # print(f"action: {action}")
                # print(f"output: {output}")
            obs, _, dones, info = env.step([action+1])
            # import pdb; pdb.set_trace()
            if kwargs.get('verbose', False):
                pass
            # print(f"info: {info}")
            if dones:
                if info[0]["solved"]:
                    # import pdb; pdb.set_trace()
                    solved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))
                    reward = info[0]["reward"]
                    rewards.append(reward)
                    
                    print("Mean reward: {reward}".format(reward=sum(rewards)/len(rewards)))
                    char_map = str_map_to_char_map(info[0]["final_map"])
                    results["solved_maps"].append(char_map)
                    results["num_boostrap_episodes"].append(0)
                    results["num_ppo_timesteps"].append(0)
                    results["train_process"].append(0)
                    results["num_boostrap_epochs"].append(0)
                    results["bootstrap_total_time"].append(0)
                    results["ppo_total_time"].append(0)
                    results["total_train_time"].append(0)
                else:
                    unsolved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))
                break
        dones = False
        obs = env.reset()
        # os.system(f"chmod 777 {experiment_path}/inference_results.json".format(experiment_path=kwargs['experiment_path']))
        # with open(kwargs['experiment_path']+"/inference_results.json", "w") as f:
        #     f.write(json.dumps(results))
        # time.sleep(0.2)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Inference script',
        description='This is the inference script for evaluating agent performance'
    )
    parser.add_argument('--game', '-g', choices=['zelda', 'loderunner'], default="zelda") 
    parser.add_argument('--representation', '-r', default='narrow')
    parser.add_argument('--results_path', default="ppo_100M_steps")
    parser.add_argument('--experiment', default="1", type=str)
    parser.add_argument('--chg_pct', default=3.0, type=float)
    parser.add_argument('--trials', default=500, type=int)
    parser.add_argument('--verbose', default=True, type=bool)

    return parser.parse_args()


################################## MAIN ########################################
if __name__ == '__main__':
    args = parse_args()
    game = args.game 
    representation = args.representation
    
    experiment_path = "./experiments/" + args.game + "/" + args.experiment + "/inference_results"
    experiment_filepath = pathlib.Path(experiment_path)
    if not experiment_filepath.exists():
        os.makedirs(str(experiment_filepath))

    
    kwargs = {
        'change_percentage': args.chg_pct,
        'trials': args.trials,
        'verbose': args.verbose,
        'experiment_path': experiment_path
    }
    
    
    infer(game, representation, args.experiment, **kwargs)