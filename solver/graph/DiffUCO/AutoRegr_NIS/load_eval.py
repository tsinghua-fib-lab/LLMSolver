import os
import pickle

import math

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import argparse
from Network.AutoregressiveNN import AutoregressiveNN
from Jraph_creator.JraphCreator import create_graph
from ImportanceSampler import ImportanceSamplerClass as IS
import wandb

parser = argparse.ArgumentParser()

parser.add_argument('--wandb_ids', default=["np8r5ikl"], type = str, nargs = "+")
parser.add_argument('--seeds', default=3, type = int)

args = parser.parse_args()


if(__name__ == "__main__"):
    for wandb_id in args.wandb_ids:

        seed_list = []
        seed_list_2 = []

        for seed in range(args.seeds):
            print("initialize ANN")
            ann = AutoregressiveNN(grid_size = 1)

            res = ann.load_dict(wandb_id, seed)
            res2 = ann.load_dict(wandb_id, seed, dict_name = "MCMC")
            print(res)
            seed_list.append(res)
            seed_list_2.append(res2)


        one_dict = seed_list[0]

        keys = one_dict.keys()

        for key in keys:
            print(key, np.mean([dict[key][-1] for dict in seed_list]), np.std([dict[key][-1] for dict in seed_list])/np.sqrt(args.seeds))

        MCMC_list = [el["MCMC_energy"][-1] for el in seed_list_2]
        print("MCMC_energy", np.mean(MCMC_list), np.std(MCMC_list)/np.sqrt(args.seeds))
