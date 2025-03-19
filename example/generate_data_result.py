import os
import json
import torch
from tqdm import tqdm
from tensordict.tensordict import TensorDict
from rl4co.data.utils import load_npz_to_tensordict
from solver.solver_pool import SolverPool
from envs.mtvrp import MTVRPEnv, MTVRPGenerator

problem_type_dict = {
    'cvrp': 'cvrp',
    'ovrp': 'ovrp',
    'vrpb': 'vrpb',
    'cvrpl': 'vrpl',
    'cvrptw': 'vrptw'
}
if __name__ == '__main__':
    policy_dir = f'/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver/solver/model_checkpoints/100/'
    lkh_path = f'/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver/solver/lkh_solver/LKH-3.0.13/LKH'
    lkh_num_runs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver_pool = SolverPool(lkh_path=lkh_path, lkh_num_runs=lkh_num_runs,
                             policy_dir=policy_dir, device=device)
    folder_path = '/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver/td/'
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    result_dict = {}
    for file in tqdm(sorted(files)):
        if not file.endswith('.npz'):
            continue
        result_dict[file] = {}
        file_path = os.path.join(folder_path, file)
        problem_type = problem_type_dict[file.split('_')[0]]
        generator = MTVRPGenerator(num_loc=30, variant_preset=problem_type)
        env = MTVRPEnv(generator, check_solution=False)
        td_data = load_npz_to_tensordict(file_path)
        td_test = env.reset(td_data)
        for solver_name in ['rf-transformer']:
            actions = solver_pool.solve(td_test.clone(), solver_name=solver_name)
            rewards = env.get_reward(td_test.clone().cpu(), actions.clone().cpu())
            result_dict[file][f'{solver_name}-action'] = actions.tolist()[0]
            result_dict[file][f'{solver_name}-reward'] = rewards.tolist()[0]

    result_path = os.path.join(folder_path, 'result.json')
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
