import numpy as np

import networkx as nx
import random
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import os
import tqdm

from envs.graph.RB_graphs import generate_xu_instances
from envs.graph.utils import run_command_with_live_output


class GraphSampler(ABC):
    def __init__(self, min_n, max_n):
        self.min_n = min_n
        self.max_n = max_n

    @abstractmethod
    def generate_graph(self):
        pass


class ErdosRenyi(GraphSampler):
    def __init__(self, min_n, max_n, p):
        super().__init__(min_n, max_n)
        self.p = p

    def __str__(self):
        return f"ER_{self.min_n}_{self.max_n}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.erdos_renyi_graph(n, self.p)


class BarabasiAlbert(GraphSampler):
    def __init__(self, min_n, max_n, m):
        super().__init__(min_n, max_n)
        self.m = m

    def __str__(self):
        return f"BA_{self.min_n}_{self.max_n}_{self.m}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.barabasi_albert_graph(n, min(self.m, n))

class RBDatasetGenerator(GraphSampler):
    """
	Class for generating datasets with RB Graphs
	"""
    def __init__(self, min_n, max_n):
        super().__init__(min_n, max_n)
        self.dataset_name = 'small'
        self.graph_config = self.__init_graph_config(self.dataset_name)


    def __str__(self):
        return f"RB_{self.dataset_name}"

    def __init_graph_config(self, dataset_name):
        """
        :param dataset_name: dataset name containing the size of the dataset
        :return: graph_config: parameter config needed to generate graph instances
        """
        if "small" in dataset_name:
            self.size = "small"
            graph_config = {
                "p_low": 0.3, "p_high": 1,
                "n_min": self.min_n, "n_max": self.max_n,
                "n_low": 15, "n_high": 20,
                "k_low": 3, "k_high": 8,
                "n_train": 4000, "n_val": 500, "n_test": 1000
            }
        elif "large" in dataset_name:
            self.size = "large"
            graph_config = {
                "p_low": 0.3, "p_high": 1,
                "n_min": 800, "n_max": 1200,
                "n_low": 40, "n_high": 55,
                "k_low": 20, "k_high": 25,
                "n_train": 4000, "n_val": 500, "n_test": 1000
            }
        elif "100" in dataset_name:
            self.size = "100"
            graph_config = {
                "p_low": 0.25, "p_high": 1,
                "n_min": 0, "n_max": np.inf,
                "n_low": 9, "n_high": 15,
                "k_low": 8, "k_high": 11,
                "n_train": 3000, "n_val": 500, "n_test": 1000
            }
        elif "200" in dataset_name:
            self.size = "200"
            graph_config = {
                "p_low": 0.25, "p_high": 1,
                "n_min": 0, "n_max": np.inf,
                "n_low": 20, "n_high": 25,
                "k_low": 9, "k_high": 10,
                "n_train": 2000, "n_val": 500, "n_test": 1000
            }
        elif "huge" in dataset_name:
            self.size = "1000"
            graph_config = {
                "p_low": 0.25, "p_high": 1,
                "n_min": 0, "n_max": np.inf,
                "n_low": 60, "n_high": 70,
                "k_low": 15, "k_high": 20,
                "n_train": 3000, "n_val": 500, "n_test": 1000
            }
        elif "giant" in dataset_name:
            self.size = "2000"
            graph_config = {
                "p_low": 0.25, "p_high": 1,
                "n_min": 0, "n_max": np.inf,
                "n_low": 120, "n_high": 140,
                "k_low": 15, "k_high": 20,
                "n_train": 3000, "n_val": 500, "n_test": 1000
            }
        elif "dummy" in dataset_name:
            self.size = "dummy"
            graph_config = {
                "p_low": 0.25, "p_high": 1,
                "n_min": 0, "n_max": np.inf,
                "n_low": 9, "n_high": 15,
                "k_low": 8, "k_high": 11,
                "n_train": 300, "n_val": 500, "n_test": 1000
            }
        else:
            raise NotImplementedError('Dataset name must contain either "small", "large", "huge", "giant", "100", "200" to infer the number of nodes')
        return graph_config

    def generate_graph(self):

        p = np.random.uniform(self.graph_config["p_low"], self.graph_config["p_high"])
        min_n, max_n = self.graph_config["n_min"], self.graph_config["n_max"]
        n = np.random.randint(self.graph_config["n_low"], self.graph_config["n_high"])
        k = np.random.randint(self.graph_config["k_low"], self.graph_config["k_high"])

        edges = generate_xu_instances.get_random_instance(n, k, p)
        G = nx.Graph()
        G.add_edges_from(edges)
        num_nodes = G.number_of_nodes()

        return G

class HolmeKim(GraphSampler):
    def __init__(self, min_n, max_n, m, p):
        super().__init__(min_n, max_n)

        self.m = m
        self.p = p

    def __str__(self):
        return f"HK_{self.min_n}_{self.max_n}_{self.m}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.powerlaw_cluster_graph(n, min(self.m, n), self.p)


class WattsStrogatz(GraphSampler):
    def __init__(self, min_n, max_n, k, p):
        super().__init__(min_n, max_n)

        self.k = k
        self.p = p

    def __str__(self):
        return f"WS_{self.min_n}_{self.max_n}_{self.k}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.watts_strogatz_graph(n, self.k, self.p)


class HyperbolicRandomGraph(GraphSampler):
    def __init__(self, min_n, max_n, alpha, t, degree, threads):
        super().__init__(min_n, max_n)

        self.alpha = alpha
        self.t = t
        self.degree = degree
        self.threads = threads

        girgs_path = Path(__file__).parent / "girgs"

        if not girgs_path.exists():
            girgs_repo = "https://github.com/chistopher/girgs"
            target_commit = "c38e4118f02cffae51b1eaf7a1c1f9314a6a89c8"
            subprocess.run(["git", "clone", girgs_repo], cwd=Path(__file__).parent)
            subprocess.run(["git", "checkout", target_commit], cwd=girgs_path)
            os.mkdir(girgs_path / "build")
            subprocess.run(["cmake", ".."], cwd=girgs_path / "build")
            subprocess.run(["make", "genhrg"], cwd=girgs_path / "build")

        self.binary_path = girgs_path / "build" / "genhrg"
        self.tmp_path = girgs_path

    def __str__(self):
        return f"HRG_{self.min_n}_{self.max_n}_{self.alpha}_{self.t}_{self.degree}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        command = [self.binary_path, "-n", str(n), "-alpha", str(self.alpha), "-t", str(self.t), "-deg",
                   str(self.degree), "-threads", str(self.threads), "-edge", "1", "-file", str(self.tmp_path / "tmp")]
        run_command_with_live_output(command)

        with open(self.tmp_path / "tmp.txt", 'r') as file:
            content = file.read().split('\n')

        edge_list = list(map(lambda x: tuple(map(int, x.split())), content[2:-1]))
        G = nx.empty_graph(n)
        G.add_edges_from(edge_list)
        os.remove(self.tmp_path / "tmp.txt")

        return G


class RandomGraphGenerator:
    def __init__(self, graph_sampler: GraphSampler, num_graphs=1):
        self.num_graphs = num_graphs
        self.graph_sampler = graph_sampler

    def random_weight(self, n, mu=1, sigma=0.1):
        return np.around(np.random.normal(mu, sigma, n)).astype(int).clip(min=0)

    def generate(self, weighted=False):
        generated_graph = []
        for i in tqdm.tqdm(range(self.num_graphs)):

            G = self.graph_sampler.generate_graph()
            G.remove_nodes_from(list(nx.isolates(G)))

            if weighted:
                for u, v in G.edges():
                    G[u][v]['weight'] = random.random()

            generated_graph.append(G)

        return generated_graph


if __name__ == '__main__':
    pass
