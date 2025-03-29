import os
import re
import json
import time
from typing import List

import numpy as np
from tqdm import tqdm

from LLM import LLM_api
from envs import MTVRPGenerator, MTVRPEnv
from seed_cvrp import cvrp_data_seed
from seed_ovrp import ovrp_data_seed
from seed_vrpb import vrpb_data_seed
from seed_vrpmb import vrpmb_data_seed
from seed_vrptw import vrptw_data_seed


def get_user_template_data(data_template_dict: dict, tags: list) -> dict:
    """Extract user template data from the given dictionary"""
    user_template_dict = {}
    for tag in tags:
        if tag == 'num_customer':
            user_template_dict[tag] = [len(data_template_dict['locs'])]
        elif tag == 'num_depot':
            user_template_dict[tag] = [data_template_dict['num_depot'][0]]
        elif tag == 'locs':
            user_template_dict[tag] = data_template_dict['locs']
        elif tag == 'loc_depot':
            user_template_dict[tag] = data_template_dict['locs'][0]
        elif tag == 'loc_customer':
            user_template_dict[tag] = data_template_dict['locs'][1:]
        elif tag == 'demand' or tag == 'demand_linehaul':
            user_template_dict[tag] = data_template_dict['demand_linehaul']
        elif tag == 'demand_backhaul':
            user_template_dict[tag] = data_template_dict['demand_backhaul']
        elif tag == 'capacity':
            user_template_dict[tag] = data_template_dict['vehicle_capacity']
        elif tag == 'speed':
            user_template_dict[tag] = data_template_dict['speed']
        elif tag == 'service_time':
            user_template_dict[tag] = data_template_dict['service_time']
        elif tag == 'depot_service_time':
            user_template_dict[tag] = data_template_dict['service_time'][0]
        elif tag == 'time_windows':
            user_template_dict[tag] = data_template_dict['time_windows']
        elif tag == 'time_windows_open':
            user_template_dict[tag] = [[row[0]] for row in data_template_dict['time_windows']]
        elif tag == 'time_windows_close':
            user_template_dict[tag] = [[row[1]] for row in data_template_dict['time_windows']]
        elif tag == 'depot_open_time':
            user_template_dict[tag] = data_template_dict['time_windows'][0][0]
        elif tag == 'depot_close_time':
            user_template_dict[tag] = data_template_dict['time_windows'][0][1]
        elif tag == 'locs_time_windows':
            user_template_dict = np.hstack(
                [np.array(data_template_dict['logs']), np.array(data_template_dict['time_windows'])]).tolist()
        else:
            raise NotImplementedError(f"Tag {tag} not implemented")
    return user_template_dict


def generate_data(problem_type):
    num_loc = np.random.randint(10, 30)
    variant_preset = problem_type
    if problem_type == 'vrpmb':
        variant_preset = 'vrpb'
    generator = MTVRPGenerator(num_loc=num_loc, variant_preset=variant_preset)
    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(1)
    td_data_dict = {k: v.numpy()[0].tolist() for k, v in td_data.items()}
    if problem_type == 'vrpmb':
        td_data_dict['backhaul_class'] = [2]
    return td_data_dict


if __name__ == '__main__':
    problem_type = 'cvrp'
    generated_problem_type_path = f"./data/{problem_type}_meta.json"
    with open(generated_problem_type_path, "r") as f:
        scenario_list = json.load(f)
    scenario_data_list = []
    for scenario in scenario_list:
        # Generate data for each scenario
        data_dict = generate_data(problem_type)
        scenario_data = {
            "title": scenario["title"],
            "desc_split": scenario["content"],
            "data_template": data_dict,
            "user_template": get_user_template_data(data_dict, scenario["tags"]),
            "label": problem_type
        }
        scenario_data_list.append(scenario_data)
    scenario_problem_type_path = f"./data/{problem_type}.json"
    with open(scenario_problem_type_path, "w") as f:
        json.dump(scenario_data_list, f, indent=4)
