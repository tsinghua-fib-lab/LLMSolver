import json

import numpy as np
from envs.mtdvrp import MTVRPGenerator, MTVRPEnv


def get_user_template_data(data_template_dict: dict, tags: list) -> dict:
    """Extract user template data from the given dictionary"""
    user_template_dict = {}
    for tag in tags:
        if tag == 'num_customer':
            user_template_dict[tag] = [len(data_template_dict['locs'])]
        elif tag in ['num_depots', 'num_depot']:
            user_template_dict[tag] = [data_template_dict['num_depots'][0]]
        elif tag == 'locs':
            user_template_dict[tag] = data_template_dict['locs']
        elif tag == 'loc_depot':
            num_depots = data_template_dict['num_depots'][0]
            user_template_dict[tag] = data_template_dict['locs'][:num_depots]
        elif tag == 'loc_customer':
            num_depots = data_template_dict['num_depots'][0]
            user_template_dict[tag] = data_template_dict['locs'][num_depots:]
        elif tag == 'loc_backhaul':
            nonzero_indices = np.where(np.array(data_template_dict['demand_backhaul']) > 0)[0]
            num_depots = data_template_dict['num_depots'][0]
            locs_backhaul = np.array(data_template_dict['locs'][num_depots:])
            user_template_dict[tag] = locs_backhaul[nonzero_indices].tolist()
        elif tag in ['demand', 'demand_linehaul', 'linehaul_demand']:
            user_template_dict[tag] = data_template_dict['demand_linehaul']
        elif tag in ['demand_backhaul', 'backhaul_demand']:
            user_template_dict[tag] = data_template_dict['demand_backhaul']
        elif tag in ['capacity', 'C']:
            user_template_dict[tag] = data_template_dict['vehicle_capacity']
        elif tag == 'speed':
            user_template_dict[tag] = data_template_dict['speed']
        elif tag == 'service_time':
            user_template_dict[tag] = data_template_dict['service_time']
        elif tag == 'depot_service_time':
            num_depots = data_template_dict['num_depots'][0]
            user_template_dict[tag] = data_template_dict['service_time'][:num_depots]
        elif tag in ['time_window', 'time_windows', "TW"]:
            user_template_dict[tag] = data_template_dict['time_windows']
        elif tag in ['time_windows_open', 'time_window_start', 'time_windows_start', 'time_window_open']:
            user_template_dict[tag] = [[row[0]] for row in data_template_dict['time_windows']]
        elif tag in ['time_windows_close', 'time_window_end', 'time_windows_end', 'time_window_close']:
            user_template_dict[tag] = [[row[1]] for row in data_template_dict['time_windows']]
        elif tag == 'depot_open_time':
            num_depots = data_template_dict['num_depots'][0]
            user_template_dict[tag] = [row[0] for row in data_template_dict['time_windows'][:num_depots]]
        elif tag in ['depot_close_time']:
            num_depots = data_template_dict['num_depots'][0]
            user_template_dict[tag] = [row[1] for row in data_template_dict['time_windows'][:num_depots]]
        elif tag == 'locs_time_windows':
            user_template_dict = np.hstack(
                [np.array(data_template_dict['logs']), np.array(data_template_dict['time_windows'])]).tolist()
        elif tag in ['distance_limit', 'L']:
            user_template_dict[tag] = data_template_dict['distance_limit']
        elif tag in ['time_limit', 'duration', 'duration_limit', 'duration_limits', 'hour_limit', 'hourly_limit']:
            user_template_dict[tag] = [data_template_dict['distance_limit'][0] / data_template_dict['speed'][0]]
        else:
            raise NotImplementedError(f"Tag {tag} not implemented")
    return user_template_dict


def generate_data(problem_type):
    num_depots = 1
    num_loc = np.random.randint(10, 30)
    backhaul_class = 1
    if 'md' in problem_type:
        problem_type = problem_type.replace('md', '')
        num_depots = np.random.randint(2, 5)
    if 'mb' in problem_type:
        problem_type = problem_type.replace('mb', 'b')
        backhaul_class = 2
    generator = MTVRPGenerator(num_depot=num_depots,
                               num_loc=num_loc,
                               backhaul_class=backhaul_class,
                               variant_preset=problem_type)
    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(1)
    td_data_dict = {k: v.numpy()[0].tolist() for k, v in td_data.items()}
    return td_data_dict


problem_type_base = ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw', 'vrpmb', 'mdcvrp']

var_problem_type_list = ['ovrptw', 'ovrpb', 'ovrpl', 'vrpbl', 'vrpbtw', 'vrpltw', 'ovrpbl', 'ovrpbtw', 'ovrpltw',
                         'vrpbltw', 'ovrpbltw', 'ovrpmb', 'vrpmbl', 'vrpmbtw', 'ovrpmbl', 'ovrpmbtw', 'vrpmbltw',
                         'ovrpmbltw',

                         'mdovrp', 'mdvrpb', 'mdvrpl', 'mdvrptw', 'mdovrptw', 'mdovrpb', 'mdovrpl', 'mdvrpbl',
                         'mdvrpbtw', 'mdvrpltw', 'mdovrpbl', 'mdovrpbtw', 'mdovrpltw', 'mdvrpbltw', 'mdovrpbltw',
                         'mdvrpmb', 'mdovrpmb', 'mdvrpmbl', 'mdvrpmbtw', 'mdovrpmbl', 'mdovrpmbtw', 'mdvrpmbltw',
                         'mdovrpmbltw']

if __name__ == '__main__':
    problem_type_list = var_problem_type_list
    problem_type_dir = "data_var"
    for problem_type in problem_type_list:
        print(problem_type)

        generated_problem_type_path = f"./{problem_type_dir}/{problem_type}_meta.json"
        with open(generated_problem_type_path, "r") as f:
            scenario_list = json.load(f)
        scenario_data_list = []
        for scenario in scenario_list:
            # Generate data for each scenario
            data_dict = generate_data(problem_type)
            print(scenario["title"])
            scenario_data = {
                "title": scenario["title"],
                "desc_split": scenario["content"],
                "data_template": data_dict,
                "user_template": get_user_template_data(data_dict, scenario["tags"]),
                "label": problem_type
            }
            scenario_data_list.append(scenario_data)
        scenario_problem_type_path = f"./{problem_type_dir}/{problem_type}.json"
        with open(scenario_problem_type_path, "w") as f:
            json.dump(scenario_data_list, f, indent=4)
