import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

from envs.graph.env import GraphEnv
from envs.graph.generator import GraphGenerator
from solver.graph.DiffUCO.diffuco_solver import DiffUCOSolver
from solver.graph.FastT2T.fastT2T_solver import FastT2TSolver
from solver.graph.Gurobi.gurobi_solver import GurobiSolver


def test_solve():
    # problem_type = 'maxcut'
    problem_type = 'mis'
    # problem_type = 'mvc'
    # problem_type = 'mds'
    generator = GraphGenerator(problem_type=problem_type)
    grapg_env = GraphEnv(problem_type=problem_type)

    instances = generator.generate(10)
    groubi_solver = GurobiSolver(problem_type)
    groubi_solutions = groubi_solver.solve(instances)
    print(groubi_solutions)
    groubi_rewards = []
    for instance, solution in zip(instances, groubi_solutions):
        reward = grapg_env.get_reward(instance, solution)
        groubi_rewards.append(reward)
    print('groubi: ', np.mean(groubi_rewards))

    diffuco_solver = DiffUCOSolver(problem_type)
    diffuco_solutions = diffuco_solver.solve(instances)

    print(diffuco_solutions)
    diffuco_rewards = []
    for instance, solution in zip(instances, diffuco_solutions):
        reward = grapg_env.get_reward(instance, solution)
        diffuco_rewards.append(reward)
    print('diffuco: ', np.mean(diffuco_rewards))

    if problem_type == 'mis':
        fast_t2t_solver = FastT2TSolver(problem_type=problem_type)
        node_solutions = fast_t2t_solver.solve(instances, )
        print(node_solutions)
        rewards = []
        for instance, solution in zip(instances, node_solutions):
            reward = grapg_env.get_reward(instance, solution)
            rewards.append(reward)
        print('fast_t2t: ', np.mean(rewards))


if __name__ == '__main__':
    test_solve()
