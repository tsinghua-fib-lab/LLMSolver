import networkx as nx
from typing import Iterable, Set, Hashable, Union

from envs.graph.generator import data2graph

Node = Hashable


def _ensure_set(sol: Iterable[Node]) -> Set[Node]:
    """Return a *set* copy of the solution, raising if duplicate or non‑hashable."""
    try:
        sol_set = set(sol)
    except TypeError as e:
        raise TypeError("Solution must contain hashable node identifiers") from e
    return sol_set


def _validate_nodes_exist(G: nx.Graph, nodes: Set[Node]):
    missing = nodes - set(G.nodes)
    if missing:
        raise ValueError(f"Solution contains nodes not in the graph: {missing}")


class GraphEnv:
    def __init__(self, problem_type: str):
        assert problem_type in ["maxcut", "mds", "maxclique", "mis", "mvc"]
        self.problem_type = problem_type

    def get_reward(self,
                   instances: dict,
                   solution: Union[Iterable[Node], dict[Node, int]],
                   ):
        """Compute the *reward* for a given solution to a classic graph‑optimization problem.

        Parameters
        ----------
        instances : dict
            The graph instance on which the problem is defined.
        solution : iterable or dict
            Representation varies with the problem type:
            * **MaxCut** – iterable of nodes representing *one* side of the cut *or* a
              dict mapping node -> {0,1} partition labels.
            * **MDS** (minimum dominating set) – iterable of dominating nodes.
            * **MaxClique** – iterable of nodes that are claimed to form a *clique*.
            * **MIS** (maximum independent set) – iterable of nodes claimed to form an
              independent set.
            * **MVC** (minimum vertex cover) – iterable of nodes claimed to cover all
              edges.

        Returns
        -------
        int | float
            The *reward* – for maximisation problems this is the objective value; for
            minimisation problems it is *negative* objective value so that larger is
            always better. Invalid solutions return ``float('-inf')``.
        """
        G = data2graph(instances)

        if self.problem_type == "maxcut":
            side_a = _ensure_set(solution)
            _validate_nodes_exist(G, side_a)
            side_b = set(G.nodes) - side_a
            cut_size = sum(1 for u, v in G.edges if (u in side_a) ^ (v in side_a))
            return cut_size

        elif self.problem_type == "mds":  # *minimise* size
            dom = _ensure_set(solution)
            _validate_nodes_exist(G, dom)
            # every node must be dominated
            for v in G:
                if v not in dom and not any(u in dom for u in G.neighbors(v)):
                    return float("-inf")  # invalid dominating set
            return -len(dom)

        elif self.problem_type == "maxclique":
            clique = _ensure_set(solution)
            _validate_nodes_exist(G, clique)
            # check clique
            for u in clique:
                for v in clique:
                    if u == v:
                        continue
                    if not G.has_edge(u, v):
                        return float("-inf")
            return len(clique)

        elif self.problem_type == "mis":  # maximise size
            indep = _ensure_set(solution)
            _validate_nodes_exist(G, indep)
            # check no edge internal
            for u in indep:
                if any(v in indep for v in G.neighbors(u)):
                    return float("-inf")
            return len(indep)

        elif self.problem_type == "mvc":  # *minimise* size
            cover = _ensure_set(solution)
            _validate_nodes_exist(G, cover)
            for u, v in G.edges:
                if u not in cover and v not in cover:
                    return float("-inf")
            return -len(cover)

        else:
            raise ValueError("Unsupported problem type; choose from MaxCut, MDS, MaxClique, MIS, MVC.")

    def check_valid(self,
                    instances: dict,
                    solution: Union[Iterable[Node], dict[Node, int]], ):
        G = data2graph(instances)

        if self.problem_type == "maxcut":
            try:
                side_a = _ensure_set(solution)
                _validate_nodes_exist(G, side_a)
            except Exception:
                return False
        elif self.problem_type == "mds":  # *minimise* size
            try:
                dom = _ensure_set(solution)
                _validate_nodes_exist(G, dom)
            except Exception:
                return False
            # every node must be dominated
            for v in G:
                if v not in dom and not any(u in dom for u in G.neighbors(v)):
                    return False  # invalid dominating set

        elif self.problem_type == "maxclique":
            try:
                clique = _ensure_set(solution)
                _validate_nodes_exist(G, clique)
            except Exception:
                return False
            # check clique
            for u in clique:
                for v in clique:
                    if u == v:
                        continue
                    if not G.has_edge(u, v):
                        return False

        elif self.problem_type == "mis":  # maximise size
            try:
                indep = _ensure_set(solution)
                _validate_nodes_exist(G, indep)
            except Exception:
                return False
            # check no edge internal
            for u in indep:
                if any(v in indep for v in G.neighbors(u)):
                    return False

        elif self.problem_type == "mvc":  # *minimise* size
            try:
                cover = _ensure_set(solution)
                _validate_nodes_exist(G, cover)
            except Exception:
                return False
            for u, v in G.edges:
                if u not in cover and v not in cover:
                    return False
        else:
            raise ValueError("Unsupported problem type; choose from MaxCut, MDS, MaxClique, MIS, MVC.")

        return True
