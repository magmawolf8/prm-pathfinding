import heapq
from typing import List, Dict, Tuple, Set

from configurations import ConfigSpace


def reconstruct_path(preceding: Dict[int, int], node: int) -> List[int]:
    """
    Reconstructs the path from start to the goal by following parent links in 'preceding'.
    """
    path = [node]
    while node in preceding:
        node = preceding[node]
        path.append(node)
    return list(reversed(path))

def a_star(net, cs: ConfigSpace, start: int, goal: int) -> List[int]:
    """
    Executes an A* search to find a path from 'start' to 'goal' in the given 'net'.

    :param net: A Network object with 'nodes' and 'n_adj' properties.
    :param cs: The ConfigSpace object with a 'distance' method.
    :param start: Index of the start node in net.nodes
    :param goal: Index of the goal node in net.nodes
    :return: A list of node indices describing the path from start to goal
    :raises Exception: If no solution is found
    """
    closed_set: Set[int] = set()
    preceding: Dict[int, int] = {}
    g: Dict[int, float] = {start: 0.0}

    # Heuristic: Straight-line distance
    r_s, c_s, a_s = net.nodes[start]
    r_g, c_g, a_g = net.nodes[goal]
    h_start = ConfigSpace.distance(r_s, c_s, a_s, r_g, c_g, a_g)
    f: Dict[int, float] = {start: h_start}

    frontier: List[Tuple[float, int]] = []
    heapq.heappush(frontier, (f[start], start))

    while frontier:
        _, curr_node = heapq.heappop(frontier)

        if curr_node in closed_set:
            continue
        if curr_node == goal:
            return reconstruct_path(preceding, curr_node)

        closed_set.add(curr_node)

        for edge_dist, neigh_idx in net.n_adj[curr_node]:
            tentative_g = g[curr_node] + edge_dist

            # If already visited with a cheaper cost, skip
            if neigh_idx in closed_set and tentative_g >= g.get(neigh_idx, float('inf')):
                continue

            if tentative_g < g.get(neigh_idx, float('inf')):
                preceding[neigh_idx] = curr_node
                g[neigh_idx] = tentative_g

                r_n, c_n, a_n = net.nodes[neigh_idx]
                h_n = ConfigSpace.distance(r_n, c_n, a_n, r_g, c_g, a_g)
                f[neigh_idx] = tentative_g + h_n

                heapq.heappush(frontier, (f[neigh_idx], neigh_idx))

    raise Exception("No solution found by A*.")