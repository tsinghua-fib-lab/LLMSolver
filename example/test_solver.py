import torch

from solver.solver_pool import SolverPool

if __name__ == '__main__':
    from envs.mtvrp import MTVRPEnv, MTVRPGenerator

    model_type = 'rf-transformer'
    policy_dir = f'/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver/solver/model_checkpoints/100/'
    lkh_path = f'/data1/zsf/Workplace/LLMSolver_Poject/LLMSolver/solver/lkh_solver/LKH-3.0.13/LKH'
    lkh_num_runs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver_pool = SolverPool(lkh_path=lkh_path, lkh_num_runs=lkh_num_runs,
                             policy_dir=policy_dir, device=device)

    # problem_type = "cvrp"
    for problem_type in ['cvrp', 'ovrp', 'vrpb', 'vrpl', 'vrptw']:
        print(problem_type)
        generator = MTVRPGenerator(num_loc=30, variant_preset=problem_type)
        env = MTVRPEnv(generator, check_solution=False)
        td_data = env.generator(1)
        td_test = env.reset(td_data)
        for solver_name in ['rf-transformer', 'lkh', 'pyvrp', 'ortools']:
            actions = solver_pool.solve(td_test.clone(), solver_name=solver_name)
            rewards = env.get_reward(td_test.clone().cpu(), actions.clone().cpu())
            print(solver_name)
            print(actions)
            print(rewards)
