import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from tensordict.tensordict import TensorDict
from rl4co.data.utils import load_npz_to_tensordict
from solver.solver_pool import SolverPool
from envs.mtvrp import MTVRPEnv, MTVRPGenerator


def eval_problem_type_soler(folder_path, problem_type, solver_name):
    generator = MTVRPGenerator(num_loc=50, variant_preset=problem_type_dict[problem_type])
    env = MTVRPEnv(generator, check_solution=False)
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    files_filtered = []
    for file in files:
        if not file.endswith('.npz'):
            continue
        if problem_type != file.split('_')[0]:
            continue
        files_filtered.append(file)
    score_dict = {}

    pbar = tqdm(sorted(files_filtered))
    for file in pbar:

        file_path = os.path.join(folder_path, file)
        problem_type = file.split('_')[0]
        pbar.set_description(f'{problem_type}-{solver_name}')
        td_data = load_npz_to_tensordict(file_path)
        td_test = env.reset(td_data)
        time_start = time.time()
        actions = solver_pool.solve(instances=td_test.clone(), solver_name=solver_name,
                                    problem_type=problem_type_dict[problem_type], timeout=15, num_procs=2, )
        time_end = time.time()

        try:
            env.check_solution_validity(td_test.clone().cpu(), actions.cpu())
        except:
            actions = '<SolutionInvalid>'
        score_dict[file] = {}
        if isinstance(actions, str):
            score_dict[file][f'{solver_name}-action'] = actions
            print(actions)
            continue

        rewards = env.get_reward(td_test.clone().cpu(), actions.clone().cpu())
        score_dict[file][f'{solver_name}-action'] = actions.tolist()[0]
        score_dict[file][f'{solver_name}-reward'] = rewards.tolist()[0]
        score_dict[file][f'{solver_name}-time'] = time_end - time_start

    result_path = os.path.join(folder_path, 'result', f'{problem_type}_{solver_name}.json')
    parent_dir = os.path.dirname(result_path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(score_dict, f, indent=4)


problem_type_dict = {
    'cvrp': 'cvrp',
    'ovrp': 'ovrp',
    'vrpb': 'vrpb',
    'cvrpl': 'vrpl',
    'cvrptw': 'vrptw'
}

solver_list = ['greedy', 'rf-transformer', 'lkh', 'ortools', 'pyvrp']

if __name__ == '__main__':
    policy_dir = f'/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver/solver/model_checkpoints/100/'
    lkh_path = f'/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver/solver/lkh_solver/LKH-3.0.13/LKH'
    lkh_num_runs = 10
    lkh_max_trials = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver_pool = SolverPool(lkh_path=lkh_path, lkh_num_runs=lkh_num_runs, lkh_max_trials=lkh_max_trials,
                             policy_dir=policy_dir, device=device)
    folder_path = '/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver/td/'

    for problem_type in problem_type_dict.keys():
        for solver_name in solver_list:
            eval_problem_type_soler(folder_path, problem_type, solver_name)

    for problem_type in problem_type_dict.keys():
        greedy_path = os.path.join(folder_path, 'result', f'{problem_type}_greedy.json')
        with open(greedy_path, 'r') as f:
            greedy_dict = json.load(f)

        for solver_name in solver_list:
            result_path = os.path.join(folder_path, 'result', f'{problem_type}_{solver_name}.json')
            with open(result_path, 'r') as f:
                result_dict = json.load(f)
            score_list = []
            time_list = []
            error_list = []
            for key in result_dict.keys():
                action = result_dict[key][f'{solver_name}-action']
                if not isinstance(action, str):
                    norm_score = result_dict[key][f'{solver_name}-reward'] / greedy_dict[key][f'greedy-reward']
                    score_list.append(norm_score)
                    time_list.append(result_dict[key][f'{solver_name}-time'])
                else:
                    error_list.append(key)
            print('-----------------------------------------------------------')
            print(f'{problem_type}-{solver_name}:')
            print(f"Avg Score: {np.mean(score_list)}")
            print(f"Avg Time: {np.mean(time_list)}")
            print(f"Avg Error: {len(error_list) / len(result_dict.keys())}")
