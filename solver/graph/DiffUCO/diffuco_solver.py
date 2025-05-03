import os
import argparse
import pickle
import jraph

import networkx as nx
import igraph as ig
import numpy as np

from envs.graph.generator import GraphGenerator, data2graph
from envs.graph.env import GraphEnv
from solver.graph.DiffUCO.ConditionalExpectation import ConditionalExpectation
from solver.graph.DiffUCO.DatasetCreator.Gurobi import GurobiSolver


##########################################################################################
# parameters

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb_id', default="114mqmhk", type=str)
    parser.add_argument('--dataset', default="BA_small", type=str)
    parser.add_argument('--GPU', default="0", type=str)
    parser.add_argument('--evaluation_factor', default=3, type=int)
    parser.add_argument('--n_samples', default=40, type=int, help="number of samples for each graph")
    parser.add_argument('--batch_size', default=1, type=int, help="number of graphs in each forward pass")
    parser.add_argument('--diff_ps', default=False, type=bool, help="")
    parser.add_argument('--measure_time', default=False, type=bool, help="")

    args = parser.parse_args()
    return args


##########################################################################################
# format


def nx_to_igraph(gnx: nx.Graph) -> ig.Graph:
    """
    Convert networkx graph to igraph graph

    :param gnx: networkx graph
    :return: igraph graph
    """
    return ig.Graph.TupleList(gnx.edges(), directed=False)


def from_igraph_to_jgraph(igraph, zero_edges=True, double_edges=True, _np=np):
    num_vertices = igraph.vcount()
    edge_arr = _np.array(igraph.get_edgelist())
    if (double_edges):
        # print("ATTENTION: edges will be dublicated in this method!")
        if (igraph.ecount() > 0):
            undir_receivers = edge_arr[:, 0]
            undir_senders = edge_arr[:, 1]
            receivers = _np.concatenate([undir_receivers[:, np.newaxis], undir_senders[:, np.newaxis]], axis=-1)
            receivers = _np.ravel(receivers)
            senders = _np.concatenate([undir_senders[:, np.newaxis], undir_receivers[:, np.newaxis]], axis=-1)
            senders = _np.ravel(senders)
            edges = _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype=np.int32)
            senders = _np.ones((0,), dtype=np.int32)
            edges = _np.ones((0, 1))

        if (not zero_edges):
            edge_weights = igraph.es["weight"]
            edges = _np.concatenate([edge_weights, edge_weights], axis=0)
    else:
        if (igraph.ecount() > 0):
            senders = edge_arr[:, 0]
            receivers = edge_arr[:, 1]
            edges = _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype=np.int32)
            senders = _np.ones((0,), dtype=np.int32)
            edges = _np.ones((0, 1))

        if (not zero_edges):
            edge_weights = igraph.es["weight"]
            edges = _np.array(edge_weights)

    nodes = _np.zeros((num_vertices, 1))
    globals = _np.array([num_vertices])
    n_node = _np.array([num_vertices])
    n_edge = _np.array([receivers.shape[0]])

    jgraph = jraph.GraphsTuple(senders=senders, receivers=receivers, edges=edges, nodes=nodes, n_edge=n_edge,
                               n_node=n_node, globals=globals)
    return jgraph


def igraph_to_jraph(g: ig.Graph) -> (jraph.GraphsTuple, float, int):
    """
    Convert igraph graph to jraph graph

    :param g: igraph graph
    :return: (H_graph, density, graph_size)
    """
    density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
    graph_size = g.vcount()
    return from_igraph_to_jgraph(g), density, graph_size


def solve_graph(H_graph, g, gurobi_solve=False, problem_type="maxcut", time_limit=1., thread_fraction=0.75) -> (float, float,
                                                                                                           list, float,
                                                                                                           jraph.GraphsTuple):
    """
    Solve the graph instance for the dataset using gurobi if self.gurobi_solve is True, otherwise return None Tuple

    :param H_graph: jraph graph instance
    :param g: igraph graph instance
    :return: (Energy, boundEnergy, solution, runtime, H_graph_compl)
    """
    if gurobi_solve:
        if problem_type == "maxcut":
            H_graph_compl = from_igraph_to_jgraph(g, double_edges=False)
            _, Energy, boundEnergy, solution, runtime, MC_value = GurobiSolver.solveMaxCut(H_graph,
                                                                                           time_limit=time_limit,
                                                                                           bnb=False, verbose=False,
                                                                                           thread_fraction=thread_fraction)
            return Energy, boundEnergy, solution, runtime, H_graph_compl

        elif problem_type == "mds":
            _, Energy, solution, runtime = GurobiSolver.solveMDS_as_MIP(H_graph, time_limit=time_limit,
                                                                        thread_fraction=thread_fraction)
            boundEnergy = Energy
            return Energy, boundEnergy, solution, runtime, None

        elif problem_type == "maxclip":
            H_graph_compl = from_igraph_to_jgraph(g.complementer(loops=False), double_edges=False)
            _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph_compl, time_limit=time_limit,
                                                                        thread_fraction=thread_fraction)
            return Energy, None, solution, runtime, H_graph_compl

        elif problem_type == "mis":
            H_graph_compl = from_igraph_to_jgraph(g, double_edges=False)
            _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph, time_limit=time_limit,
                                                                        thread_fraction=thread_fraction)
            return Energy, None, solution, runtime, H_graph_compl

        elif problem_type == "mvc":
            H_graph_compl = from_igraph_to_jgraph(g, double_edges=False)
            _, Energy, solution, runtime = GurobiSolver.solveMVC_as_MIP(H_graph, time_limit=time_limit,
                                                                        thread_fraction=thread_fraction)
            return Energy, None, solution, runtime, H_graph_compl

        else:
            raise NotImplementedError(f"Problem {problem_type} is not implemented. Choose from [MaxCut, MDS]")
    else:
        # in case gurobi is not used, arbitrary values are returned and for MaxCl, the complement graph is returned
        Energy = 0.
        boundEnergy = 0.
        solution = np.ones_like(H_graph.nodes)
        runtime = None

        if problem_type == "maxclip":
            H_graph_compl = from_igraph_to_jgraph(g.complementer(loops=False), double_edges=False)
        elif problem_type == "mis" or problem_type == "mvc" or problem_type == "maxcut":
            H_graph_compl = from_igraph_to_jgraph(g, double_edges=False)
        else:
            H_graph_compl = None
        return Energy, boundEnergy, solution, runtime, H_graph_compl


def format_instance(instances: list, gurobi_solve=False, problem="maxcut", time_limit=1., thread_fraction=0.75):
    data_list = []
    for instance in instances:
        gnx = data2graph(instance)
        edges = list(gnx.edges())
        g = ig.Graph([(edge[0], edge[1]) for edge in edges])
        isolated_nodes = [v.index for v in g.vs if v.degree() == 0]
        g.delete_vertices(isolated_nodes)
        H_graph, density, graph_size = igraph_to_jraph(g)
        Energy, boundEnergy, solution, runtime, H_graph_compl = solve_graph(H_graph, g, gurobi_solve, problem,
                                                                            time_limit, thread_fraction)
        data = {
            "Energies": Energy,
            "H_graphs": H_graph,
            "gs_bins": solution,
            "graph_sizes": graph_size,
            "densities": density,
            "runtimes": runtime,
            "upperBoundEnergies": boundEnergy,
            "compl_H_graphs": H_graph_compl,
            "time_limit": time_limit,
        }
        data_list.append(data)
    return data_list


##########################################################################################
# main


def prepare_data(problem_type, num_graphs=10, file_path=None):
    if not file_path:
        generator = GraphGenerator(problem_type=problem_type, weighted=True)
        graph_list = generator.generate(num_graphs)
        graph_list = format_instance(graph_list, gurobi_solve=True, problem=problem_type)
        return graph_list

    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        generator = GraphGenerator(problem_type=problem_type, weighted=True)
        graph_list = generator.generate(num_graphs)
        graph_list = format_instance(graph_list, gurobi_solve=True, problem=problem_type)
        with open(file_path, 'wb') as f:
            pickle.dump(graph_list, f)
        return graph_list


problemType2WandbID_dict = {
    'mis': 'm3h9mz5g', #graph_model = 'rb'
    'mvc': 'ys42lka1',  # graph_model = 'rb'
    'mds': '64dnrg5p', #graph_model = 'ba'
    'maxcut': '114mqmhk', #graph_model = 'ba'
}
problemType_dict = {
    'mis': 'MIS',
    'mds': 'MDS',
    'mvc': 'MVC',
    'maxcut': 'MaxCut',
}


class DiffUCOSolver():
    def __init__(self, problem_type: str):
        args = arg_parser()
        device = args.GPU
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.92"

        wandb_id = problemType2WandbID_dict[problem_type]
        evaluation_factor = args.evaluation_factor
        self.problem_type = problem_type
        self._config = {
            "wandb_id": wandb_id,
            "dataset": args.dataset,
            "evaluation_factor": evaluation_factor,
            "n_samples": args.n_samples,
            "diff_ps": args.diff_ps,
            "batch_size": args.batch_size,
            "measure_time": args.measure_time
        }
        self.env = GraphEnv(problem_type=problem_type)

        base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prepare_data")
        train_num_graphs=100
        train_file_path=os.path.join(base_dir, f"train_graph_list_{problem_type}_{train_num_graphs}.pkl")
        train_graph_list = prepare_data(problem_type=problem_type,
                                        num_graphs=train_num_graphs,
                                        file_path=train_file_path)
        val_num_graphs = 10
        val_file_path = os.path.join(base_dir, f"val_graph_list_{problem_type}_{val_num_graphs}.pkl")
        val_graph_list = prepare_data(problem_type=problem_type,
                                      num_graphs=val_num_graphs,
                                      file_path=val_file_path)

        test_num_graphs = 50
        test_file_path = os.path.join(base_dir, f"test_graph_list_{problem_type}_{test_num_graphs}.pkl")
        test_graph_list = prepare_data(problem_type=problem_type,
                                       num_graphs=test_num_graphs,
                                       file_path=test_file_path)
        # from solver.graph.DiffUCO.DatasetCreator.prepare_datasets import get_dataset
        # all_graph_list = get_dataset(problem=problemType_dict[self.problem_type], modes=['train', 'val', 'test'],
        #                              gurobi_solve=True)
        # train_graph_list = all_graph_list[0]
        # val_graph_list = all_graph_list[1]
        # test_graph_list = all_graph_list[2]
        self._CE = ConditionalExpectation(wandb_id=self._config["wandb_id"], config=self._config,
                                          n_eval_samples=self._config["n_samples"],
                                          k=1,
                                          eval_step_factor=self._config["evaluation_factor"],
                                          batch_size=self._config["batch_size"],
                                          train_graph_list=train_graph_list, val_graph_list=val_graph_list,
                                          test_graph_list=test_graph_list)

    def eval_on_dataset(self, test_graph_list):
        test_log_dict = self._CE.run(p=None, dataset_name=self._config["dataset"], mode="test",
                                     measure_time=self._config["measure_time"],
                                     test_graph_list=test_graph_list)
        solutions = test_log_dict['test/solutions']
        return solutions

    def solve(self, instances: list = None, **params):
        data_list = format_instance(instances, problem=self.problem_type)

        solution_results = self.eval_on_dataset(data_list)

        solutions = []
        for inst, candidate_solution in zip(instances, solution_results):
            bst_reward = float("-inf")
            bst_solution = None
            for solution in candidate_solution:
                node_solution = np.where(np.array(solution) == 1)[0].tolist()
                reward = self.env.get_reward(inst, node_solution, self.problem_type)
                if reward > bst_reward:
                    bst_reward = reward
                    bst_solution = node_solution
            solutions.append(bst_solution)

        return solutions


##########################################################################################
# test
def test_solve():
    problem_type = 'maxcut'
    # problem_type = 'mis'
    # problem_type = 'mvc'
    graph_model = 'ba'
    # from solver.graph.DiffUCO.DatasetCreator.prepare_datasets import get_dataset
    # data_list = get_dataset(problem=problemType_dict[problem_type], modes=['test'], gurobi_solve=False, time_limits=[1.])[0]
    # instances = []
    # for data in data_list:
    #     graph = data['H_graphs']
    #     edge_list = list(zip(graph.senders.tolist(), graph.receivers.tolist()))
    #
    #     instances.append({
    #         'graph': edge_list,
    #         'num_nodes': np.max(np.array(edge_list)) + 1,
    #         'num_edges': len(edge_list),
    #     })

    generator = GraphGenerator(problem_type=problem_type)
    instances = generator.generate(10)

    diffuco_solver = DiffUCOSolver(problem_type, graph_model=graph_model)
    solutions = diffuco_solver.solve(instances)
    print(solutions)


if __name__ == '__main__':
    test_solve()
