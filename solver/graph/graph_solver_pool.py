import os
from copy import deepcopy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np


class GraphSolverPool:
    solver_problem_dict = {
        'diffuco': ['mis', 'mvc', 'maxcut'],
        'fastt2t': ['mis'],
        'gurobi': ['mis', 'mvc', 'maxcut'],
    }

    def __init__(self, **kwargs):
        """
        Initializes the RL Agent with optional parameters.

        Parameters:
        ----------
        **kwargs : dict
            Arbitrary keyword arguments. Expected keys:
            - env (str or object, optional): The environment for the RL agent (e.g., "MTVRPEnv()").
            - policy_dir (str, optional): Path to the directory storing policy files. Defaults to "./model_checkpoints/100/".
            - device (str, optional): Device to run the policy on. Defaults to "cpu".
        """
        self.problem2solver_dict = {}
        self.solver_dict = {}
        self.device = kwargs.get("device", "cpu")

        for solver_name in self.solver_problem_dict.keys():
            try:
                if solver_name == "diffuco":
                    from solver.graph.DiffUCO.diffuco_solver import DiffUCOSolver
                    self.solver_dict[solver_name] = {}
                    for problem_name in self.solver_problem_dict[solver_name]:
                        self.solver_dict[solver_name][problem_name] = DiffUCOSolver(problem_type=problem_name)
                elif solver_name == "fastt2t":
                    from solver.graph.FastT2T.fastT2T_solver import FastT2TSolver
                    self.solver_dict[solver_name] = {}
                    for problem_name in self.solver_problem_dict[solver_name]:
                        self.solver_dict[solver_name][problem_name] = FastT2TSolver(problem_type=problem_name)
                elif solver_name == "gurobi":
                    from solver.graph.Gurobi.gurobi_solver import GurobiSolver
                    self.solver_dict[solver_name] = {}
                    for problem_name in self.solver_problem_dict[solver_name]:
                        self.solver_dict[solver_name][problem_name] = GurobiSolver(problem_type=problem_name)
                for problem_name in self.solver_problem_dict[solver_name]:
                    if problem_name not in self.problem2solver_dict:
                        self.problem2solver_dict[problem_name] = []
                    self.problem2solver_dict[problem_name].append(solver_name)
            except ImportError:
                print(f"WARNING: {solver_name} is not exist.")
            except Exception as e:
                print(e)

        print(f"Avaliable solvers: ")
        for problem_name in self.problem2solver_dict.keys():
            print(f"\t{problem_name}: {self.problem2solver_dict[problem_name]}")

    def get_problem_list(self) -> list | None:
        return list(self.problem2solver_dict.keys())

    def get_solver_list(self, problem_name: str) -> list | None:
        solver_list = list(self.problem2solver_dict.get(problem_name, []))
        return solver_list

    def _solve(
            self,
            instances: dict,
            solver_name: str = "diffuco",
            problem_type: str = "maxcut",
            max_runtime: float = 30,
            num_procs: int = 1,
            **kwargs,
    ) -> dict:
        """
        Solves the AnyVRP instances with PyVRP.

        Parameters
        ----------
        instances
            TensorDict containing the AnyVRP instances to solve.
        max_runtime
            Maximum runtime for the solver.
        num_procs
            Number of processers to use to solve instances in parallel.
        data_type
            Environment mode. If "mtvrp", the instance data will be converted first.
        solver_name
            The solver to use. One of ["pyvrp", "ortools", "lkh"].

        Returns
        -------
        tuple[Tensor, Tensor]
            A Tensor containing the actions for each instance and a Tensor
            containing the corresponding costs.
        """

        if solver_name not in self.solver_dict:
            raise ValueError(f"Unknown baseline solver: {solver_name}")

        _solver = self.solver_dict[solver_name][problem_type]

        results = _solver.solve(instances=instances)

        return results

    def solve(self, instances: dict, solver_name: str = "diffuco", problem_type: str = "maxcut",
              timeout: int = 30, num_procs=32, **kwargs):
        try:
            score = self._solve(instances=instances, solver_name=solver_name, problem_type=problem_type,
                                max_runtime=timeout, num_procs=num_procs, **kwargs)
        except Exception as e:
            print(e)
            return "<RuntimeError>"
        return score


def test_solve():
    from envs.graph.generator import GraphGenerator
    from envs.graph.env import GraphEnv

    from solver.graph.Gurobi.gurobi_solver import GurobiSolver
    from solver.graph.DiffUCO.diffuco_solver import DiffUCOSolver
    from solver.graph.FastT2T.fastT2T_solver import FastT2TSolver

    # problem_type = 'maxcut'
    problem_type = 'mis'
    # problem_type = 'mvc'
    # problem_type = 'mds'

    generator = GraphGenerator(problem_type=problem_type)
    grapg_env = GraphEnv(problem_type=problem_type)

    instances = generator.generate(10)

    gurobi_solver = GurobiSolver(problem_type)
    gurobi_solutions = gurobi_solver.solve(instances)
    print(gurobi_solutions)
    gurobi_rewards = []
    for instance, solution in zip(instances, gurobi_solutions):
        reward = grapg_env.get_reward(instance, solution, problem_type)
        gurobi_rewards.append(reward)
    print('gurobi: ', np.mean(gurobi_rewards))

    diffuco_solver = DiffUCOSolver(problem_type)
    diffuco_solutions = diffuco_solver.solve(instances)

    print(diffuco_solutions)
    diffuco_rewards = []
    for instance, solution in zip(instances, diffuco_solutions):
        reward = grapg_env.get_reward(instance, solution, problem_type)
        diffuco_rewards.append(reward)
    print('diffuco: ', np.mean(diffuco_rewards))

    if problem_type == 'mis':
        fast_t2t_solver = FastT2TSolver(problem_type=problem_type)
        node_solutions = fast_t2t_solver.solve(instances, )
        print(node_solutions)
        rewards = []
        for instance, solution in zip(instances, node_solutions):
            reward = grapg_env.get_reward(instance, solution, problem_type)
            rewards.append(reward)
        print('fastt2t: ', np.mean(rewards))


def test():
    from envs.graph.generator import GraphGenerator
    from envs.graph.env import GraphEnv

    # problem_type = 'maxcut'
    problem_type = 'mis'
    # problem_type = 'mvc'
    # problem_type = 'mds'

    generator = GraphGenerator(problem_type=problem_type)
    grapg_env = GraphEnv(problem_type=problem_type)

    instances = generator.generate(10)

    graph_solve_pool = GraphSolverPool()
    rewards = []
    solver_name = "diffuco"
    solutions = graph_solve_pool.solve(deepcopy(instances), solver_name=solver_name, problem_type=problem_type)
    for instance, solution in zip(instances, solutions):
        reward = grapg_env.get_reward(instance, solution, problem_type)
        rewards.append(reward)
    print(solver_name, np.mean(rewards))

    solver_name = "gurobi"
    solutions = graph_solve_pool.solve(deepcopy(instances), solver_name=solver_name, problem_type=problem_type)
    for instance, solution in zip(instances, solutions):
        reward = grapg_env.get_reward(instance, solution, problem_type)
        rewards.append(reward)
    print(solver_name, np.mean(rewards))

    if problem_type == 'mis':
        solver_name = "fastt2t"
        solutions = graph_solve_pool.solve(deepcopy(instances), solver_name=solver_name, problem_type=problem_type)
        for instance, solution in zip(instances, solutions):
            reward = grapg_env.get_reward(instance, solution, problem_type)
            rewards.append(reward)
        print(solver_name, np.mean(rewards))


if __name__ == '__main__':
    test()
