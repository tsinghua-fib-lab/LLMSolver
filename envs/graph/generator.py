import networkx as nx

from envs.graph.random_graph import RandomGraphGenerator, ErdosRenyi, BarabasiAlbert, HolmeKim, WattsStrogatz, \
    HyperbolicRandomGraph, RBDatasetGenerator


def data2graph(data: dict):
    G = nx.Graph()
    G.add_nodes_from(range(data['num_nodes']))
    G.add_edges_from(data['graph'])
    return G


problemType2graphModel_dict = {
    'maxcut': 'ba',
    'maxclique': 'rb',
    'mds': 'ba',
    'mis': 'rb',
    'mvc': 'rb',
}


class GraphGenerator:
    def __init__(self, problem_type="maxcut", weighted=False,min_n: int = 10, max_n: int = 100):
        self.graph_model = problemType2graphModel_dict[problem_type]
        self.weighted = weighted
        self.min_n = min_n
        self.max_n = max_n

    def _data_generation(self,
                         graph_model="er",
                         min_n: int = 40,
                         max_n: int = 60,
                         num_graphs: int = 1,
                         weighted=False,
                         **params):
        if graph_model == "er":
            er_p = params.get("er_p", 0.5)
            graph_generator = ErdosRenyi(min_n, max_n, er_p)
        elif graph_model == "ba":
            ba_m = params.get("ba_m", 4)
            graph_generator = BarabasiAlbert(min_n, max_n, ba_m)
        elif graph_model == "rb":
            graph_generator = RBDatasetGenerator(min_n, max_n)
        elif graph_model == "hk":
            hk_m = params.get("hk_m", 10)
            hk_p = params.get("hk_p", 0.5)
            graph_generator = HolmeKim(min_n, max_n, hk_m, hk_p)
        elif graph_model == "ws":
            ws_k = params.get("ws_k", 2)
            ws_p = params.get("ws_p", 0.5)
            graph_generator = WattsStrogatz(min_n, max_n, ws_k, ws_p)
        elif graph_model == "hrg":
            hrg_alpha = params.get("hrg_alpha", 0.75)
            hrg_t = params.get("hrg_t", 0)
            hrg_degree = params.get("hrg_degree", 2)
            hrg_threads = params.get("hrg_threads", 8)

            graph_generator = HyperbolicRandomGraph(min_n, max_n, hrg_alpha, hrg_t, hrg_degree, hrg_threads)
        else:
            raise ValueError(f"Unknown random graph model {graph_model}")
        gen = RandomGraphGenerator(graph_generator, num_graphs=num_graphs)

        graph_list = gen.generate(weighted=weighted)
        data_list = []
        for graph in graph_list:
            if weighted:
                edge_list = [
                    (int(u), int(v), {'weight': float(w['weight'])})
                    for u, v, w in graph.edges(data=weighted)
                ]
            else:
                edge_list = [
                    (int(u), int(v))
                    for u, v in graph.edges(data=weighted)
                ]
            data_list.append({
                'graph': edge_list,
                'num_nodes': int(graph.number_of_nodes()),
                'num_edges': int(graph.number_of_edges()),
            })
        return data_list

    def generate(self, batch_size) -> list[dict]:
        return self._data_generation(graph_model=self.graph_model,
                                     num_graphs=batch_size,
                                     weighted=self.weighted,
                                     min_n=self.min_n,
                                     max_n=self.max_n,)


def test_generate_grapg():
    problem_type = 'mis'
    generator = GraphGenerator(problem_type=problem_type)

    data_list = generator.generate(1)
    G = data2graph(data_list[0])
    print(G)


if __name__ == '__main__':
    test_generate_grapg()
