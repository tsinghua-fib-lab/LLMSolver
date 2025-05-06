import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tensordict.tensordict import TensorDict

from solver.solver_pool import SolverPool

if __name__ == '__main__':
    from envs.mtvrp import MTVRPEnv, MTVRPGenerator


    model_type = 'rf-transformer'
    policy_dir = f'/data1/shy/zgc/llm_solver/LLMSolver/solver/model_checkpoints/100'
    lkh_path = f'/data1/shy/zgc/llm_solver/LLMSolver/solver/lkh_solver/LKH-3.0.13/LKH'
    lkh_num_runs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver_pool = SolverPool(lkh_path=lkh_path, lkh_num_runs=lkh_num_runs,
                             policy_dir=policy_dir, device=device)
    # problem_type = "cvrp"
    for problem_type in ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw']:
        print(problem_type)
        generator = MTVRPGenerator(num_loc=30, variant_preset=problem_type)
        env = MTVRPEnv(generator, check_solution=False)
        extracted_dict =  {'locs': [[4, 12], [18, 52], [22, 38], [36, 30], [45, 60], [55, 55], [50, 35], [40, 40], [30, 50], [25, 65], [20, 20], [10, 25], [15, 45], [35, 25], [60, 45], [65, 35]], 
                           'demand_backhaul': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                           'demand_linehaul': [0.2, 0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.1, 0.2], 
                           'backhaul_class': [0], 
                           'distance_limit': [float('inf')], 
                           'time_windows': [[0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')], [0, float('inf')]], 
                           'service_time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                           'vehicle_capacity': [1], 
                           'capacity_original': [30], 
                           'open_route': [False], 
                           'speed': [1]}
        td_data =  TensorDict(
            {
                # normalize and add 1 dim at the start 
                'locs': (torch.tensor(extracted_dict['locs'])).float().unsqueeze(0),
                'demand_backhaul': torch.tensor(extracted_dict['demand_backhaul']).float().unsqueeze(0),
                'demand_linehaul': torch.tensor(extracted_dict['demand_linehaul']).float().unsqueeze(0),
                'backhaul_class': torch.tensor(extracted_dict['backhaul_class']).float().unsqueeze(0),
                'distance_limit': torch.tensor(extracted_dict['distance_limit']).float().unsqueeze(0),
                'time_windows': torch.tensor(extracted_dict['time_windows']).float().unsqueeze(0),
                'service_time': torch.tensor(extracted_dict['service_time']).float().unsqueeze(0),
                'vehicle_capacity': torch.tensor(extracted_dict['vehicle_capacity']).float().unsqueeze(0),
                'capacity_original': torch.tensor(extracted_dict['capacity_original']).float().unsqueeze(0),
                'open_route': torch.tensor(extracted_dict['open_route']).bool().unsqueeze(0),
                'speed': torch.tensor(extracted_dict['speed']).float().unsqueeze(0)
            },
            batch_size = 1
        )
        td_data = env.generator(1)
        for key in td_data.keys():
            print(key)
            print(td_data[key])
        print(td_data)
        td_test = env.reset(td_data)
        for solver_name in ['rf-transformer', 'lkh', 'pyvrp', 'ortools']:
            actions = solver_pool.solve(td_test.clone(), solver_name=solver_name)
            rewards = env.get_reward(td_test.clone().cpu(), actions.clone().cpu())
            print(solver_name)
            print(actions)
            print(rewards)