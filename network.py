import numpy as np
import pickle
from collections import defaultdict
from scipy.spatial import cKDTree

from typing import Callable, List, Tuple
from configurations import ConfigSpace


class Network:
    """
    Adjacency list implementing a probabilistic roadmap
    """

    def __init__(
            self,
            cs: ConfigSpace,
            nodes = None,
            n_adj = None
    ):
        """
        :param cs: ConfigSpace object containing the configuration space
        :param nodes: List containing the nodes in the network
        :param n_adj: Adjacency list containing each node's neighbors
        """
        self.cs = cs
        self.dim1, self.dim2, self.dim3 = cs.get_blocked_space().shape

        # PRM nodes and adjacency
        if nodes is None:
            self.nodes: List[Tuple[int, int, int]] = []
        else:
            self.nodes = nodes
        if n_adj is None:
            self.n_adj = defaultdict(list)
        else:
            self.n_adj = n_adj

        # Keep the k-d tree as None until it is explicitly built
        self.tree = None

    @staticmethod
    def load(cs: ConfigSpace, nodes_fn: str, n_adj_fn: str):
        """
        Loads a network from a binary file
        in the current working directory
        :param cs: ConfigSpace object containing the configuration space
        :param nodes_fn: the file name for the nodes file
        :param n_adj_fn: the file name for the adjacency list file
        """
        with open(nodes_fn, "rb") as nodes_f, open(n_adj_fn, "rb") as n_adj_f:
            nodes = pickle.load(nodes_f)
            n_adj = pickle.load(n_adj_f)
            return Network(cs, nodes, n_adj)

    def save(self, nodes_fn: str, n_adj_fn: str):
        """
        Saves a network as a binary file
        :param nodes_fn: the desired file name for the nodes file
        :param n_adj_fn: the desired file name for the adjacency list file
        """
        with open(nodes_fn, "wb") as nodes_f, open(n_adj_fn, "wb") as n_adj_f:
            pickle.dump(self.get_nodes(), nodes_f)
            pickle.dump(self.get_n_adj(), n_adj_f)

    def get_tree(self):
        return self.tree

    def get_nodes(self):
        return self.nodes

    def get_n_adj(self):
        return self.n_adj

    def add_node(self, row: int, col: int, angle: int) -> int:
        """
        Manually add a node to the graph
        Returns the index of the newly added node
        """
        if self.cs.get_blocked_space()[row, col, angle]:
            raise ValueError(f"Node ({row},{col},{angle}) is blocked!")
        self.nodes.append((row, col, angle))
        return len(self.get_nodes()) - 1

    def add_nodes(self, node_list: List[Tuple[int, int, int]]) -> None:
        """
        Add multiple nodes at once
        """
        for (r, c, a) in node_list:
            self.add_node(r, c, a)

    def add_random_nodes(
            self,
            how_many: int,
            pdf: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """
        Adds 'how_many' random nodes using a user-provided sampler function.

        :param how_many: number of random nodes to add
        :param pdf: a function returning the relative probability that a cell
                           with distance d from obstacles will be sampled
        """

        cdf_vals = np.cumsum(pdf(self.cs.get_edf()).ravel())
        running_sum = cdf_vals[-1]

        rng = np.random.default_rng()

        chosen = set()
        while len(chosen) < how_many:
            ind = np.searchsorted(cdf_vals, rng.random()*running_sum, side='right')
            if ind not in chosen:
                coords = ((ind // self.dim3) // self.dim2,
                          (ind // self.dim3) % self.dim2,
                          ind % self.dim3)
                if not self.cs.get_blocked_space()[coords]:
                    self.get_nodes().append(coords)
                    chosen.add(ind)

    def build_tree(self) -> None:
        """
        Builds (or rebuilds) the k-d tree for nearest-neighbor queries,
        after nodes have been added.
        """
        if not self.get_nodes():
            raise ValueError("No nodes to build a tree with.")
        self.tree = cKDTree(self.get_nodes(), boxsize=[10000, 10000, 360])

    def connect_all(self, num_neighbors: int) -> None:
        """
        Connect all nodes in the network by querying the k-d tree
        and finding the nearest neighbors for each.

        :param num_neighbors:
        """
        if self.get_tree() is None:
            raise ValueError("You must call build_tree() before connecting nodes.")
        for node_idx in range(len(self.get_nodes())):
            self._connect(node_idx, num_neighbors)

    def connect_node(self, node_idx: int, num_neighbors: int) -> None:
        """
        Connect a single node (by index) to its nearest neighbors.
        Useful if you've added one new node.
        """
        if self.get_tree() is None:
            raise ValueError("You must call build_tree() before connecting nodes.")
        self._connect(node_idx, num_neighbors)

    def _connect(self, node_num: int, num_neighbors: int) -> None:
        """
        Internal method to connect 'node_num' to its neighbors
        based on nearest neighbors in the k-d tree.
        """
        distances, indices = self.get_tree().query(self.get_nodes()[node_num], k=num_neighbors)

        # If k=1, distances and indices are scalar
        if isinstance(distances, float):
            distances = [distances]
            indices = [indices]

        for dist, neighbor_idx in zip(distances, indices):
            if neighbor_idx == node_num:
                continue
            r1, c1, a1 = self.get_nodes()[node_num]
            r2, c2, a2 = self.get_nodes()[neighbor_idx]
            # Avoid adding edges which intersect with the config space
            if not self.cs.is_blocked_edge(r1, c1, a1, r2, c2, a2):
                # Avoid double-adding the same neighbor
                if not any(nb_idx == neighbor_idx for _, nb_idx in self.get_n_adj()[node_num]):
                    self.get_n_adj()[node_num].append((dist, neighbor_idx))
                if not any(nb_idx == node_num for _, nb_idx in self.get_n_adj()[neighbor_idx]):
                    self.get_n_adj()[neighbor_idx].append((dist, node_num))