import networkx as nx
import numpy as np

from envs.graph.env import GraphEnv
from solver.graph.Gurobi.GurobiSolver import solveMaxCut, solveMDS_as_MIP, solveMIS_as_MIP, solveMVC_as_MIP


class GurobiSolver():
    def __init__(self, problem_type: str, time_limit=float("inf"), thread_fraction=0.5, num_CPUs=None):
        self.time_limit = time_limit
        self.thread_fraction = thread_fraction
        self.num_CPUs = num_CPUs
        self.problem_type = problem_type

    def solve_graph(self, instance) -> (list, float):
        """
        Solve the graph instance for the dataset using gurobi

        :return: (Energy, boundEnergy, solution, runtime, H_graph_compl)
        """
        G = nx.Graph()
        G.add_nodes_from(range(instance['num_nodes']))
        G.add_edges_from(instance['graph'])
        edge_list = []
        weight_list = []
        for u, v, d in G.edges(data=True):
            w = d.get('weight', 1.)  # 没有时给默认权重 1
            edge_list.append((u, v))
            weight_list.append(w)
            edge_list.append((v, u))
            weight_list.append(w)
        if self.problem_type == "maxcut":
            _, Energy, boundEnergy, solution, runtime, MC_value = solveMaxCut(edge_list=edge_list,
                                                                              weight_list=weight_list,
                                                                              N=instance['num_nodes'],
                                                                              time_limit=self.time_limit,
                                                                              bnb=False, verbose=False,
                                                                              thread_fraction=self.thread_fraction)
            return solution, runtime

        elif self.problem_type == "mds":
            _, Energy, solution, runtime = solveMDS_as_MIP(edge_list=edge_list,
                                                           N=instance['num_nodes'],
                                                           time_limit=self.time_limit,
                                                           thread_fraction=self.thread_fraction)
            boundEnergy = Energy
            return solution, runtime

        elif self.problem_type == "maxclique":
            graph_compl = nx.complement(G)

            edge_list = []
            weight_list = []
            for u, v, d in graph_compl.edges(data=True):
                w = d.get('weight', 1.)
                edge_list.append((u, v))
                weight_list.append(w)
                edge_list.append((v, u))
                weight_list.append(w)
            _, Energy, solution, runtime = solveMIS_as_MIP(edge_list=edge_list, N=instance['num_nodes'], time_limit=self.time_limit, thread_fraction = self.thread_fraction)
            return solution, runtime

        elif self.problem_type == "mis":
            _, Energy, solution, runtime = solveMIS_as_MIP(edge_list=edge_list,
                                                           N=instance['num_nodes'],
                                                           time_limit=self.time_limit,
                                                           thread_fraction=self.thread_fraction)
            return solution, runtime

        elif self.problem_type == "mvc":
            _, Energy, solution, runtime = solveMVC_as_MIP(edge_list=edge_list,
                                                           N=instance['num_nodes']
                                                           , time_limit=self.time_limit,
                                                           thread_fraction=self.thread_fraction)
            return solution, runtime

        else:
            raise NotImplementedError(f"Problem {self.problem_type} is not implemented. Choose from [MaxCut, MDS]")

    def solve(self, instances: list = None, **params):
        solutions = []
        for instance in instances:
            solution, runtime = self.solve_graph(instance)
            node_solution = np.where(np.array(solution) == 1)[0].tolist()
            solutions.append(node_solution)
        return solutions


##########################################################################################
# test
def test_solve():
    from envs.graph.generator import GraphGenerator

    # problem_type = 'maxcut'
    problem_type = 'maxclique'

    graph_model = 'rb'
    generator = GraphGenerator(problem_type=problem_type)
    grapg_env = GraphEnv(problem_type=problem_type)

    instances = generator.generate(10)

    groubi_solver = GurobiSolver(problem_type)
    solutions = groubi_solver.solve(instances)
    print(solutions)
    rewards = []
    for instance, solution in zip(instances, solutions):
        reward = grapg_env.get_reward(instance, solution)
        rewards.append(reward)
    print(np.mean(rewards))


if __name__ == '__main__':
    test_solve()
