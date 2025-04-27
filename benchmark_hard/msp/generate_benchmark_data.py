import json

import numpy as np

from benchmark_hard.msp.machine_scheduing_config import problem_type_param
from envs.msp.generator import SchedulingProblemType, SchedulingProblemGenerator


def get_user_template_data(data_template_dict: dict, tags: list) -> dict:
    """Extract user template data from the given dictionary"""
    user_template_dict = {}
    for tag in tags:
        if tag in data_template_dict:
            user_template_dict[tag] = data_template_dict[tag]
        elif tag == 'num_stages':
            user_template_dict['num_stages'] = len(data_template_dict['machines_per_stage'])
        else:
            raise NotImplementedError(f"Tag {tag} not implemented")
    return user_template_dict


def generate_data(problem_type):
    if problem_type == 'jssp':
        scheduling_problem_type = SchedulingProblemType.JSSP
    elif problem_type == 'fjssp':
        scheduling_problem_type = SchedulingProblemType.FJSSP
    elif problem_type == 'fssp':
        scheduling_problem_type = SchedulingProblemType.FSSP
    elif problem_type == 'hfssp':
        scheduling_problem_type = SchedulingProblemType.HFSSP
    elif problem_type == 'ossp':
        scheduling_problem_type = SchedulingProblemType.OSSP
    elif problem_type == 'asp':
        scheduling_problem_type = SchedulingProblemType.ASP
    else:
        raise NotImplementedError(f"Problem type {problem_type} not implemented")

    params = problem_type_param.get(problem_type, {})
    scheduling_generator = SchedulingProblemGenerator(scheduling_problem_type)
    scheduling_data = scheduling_generator.generate_problem_instance(**params)
    return scheduling_data


problem_type_list = ['jssp', 'fjssp', 'fssp', 'hfssp', 'ossp', 'asp']

if __name__ == '__main__':
    problem_type_dir = "data"
    for problem_type in problem_type_list:
        print(problem_type)
        generated_problem_type_path = f"./{problem_type_dir}/{problem_type}_meta.json"
        with open(generated_problem_type_path, "r") as f:
            scenario_list = json.load(f)
        scenario_data_list = []
        for scenario_idx, scenario in enumerate(scenario_list):
            # Generate data for each scenario
            data_dict = generate_data(problem_type)
            print(scenario["title"])
            scenario_data = {
                "title": scenario["title"],
                "desc_split": scenario["content"],
                "data_template": data_dict,
                "user_template": get_user_template_data(data_dict, scenario["tags"]),
                "label": problem_type,
                "index": scenario_idx,
            }
            scenario_data_list.append(scenario_data)
        scenario_problem_type_path = f"./{problem_type_dir}/{problem_type}.json"
        with open(scenario_problem_type_path, "w") as f:
            json.dump(scenario_data_list, f, indent=4)
