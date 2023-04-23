import heapq
from collections import defaultdict
from typing import Any, Literal

from disjoint_set import DisjointSet
from typing_extensions import Self

from .edge import Edge
from .node import Node


class Graph:
    """A graph data structure.

    Attributes:
        _nodes (defaultdict): A dictionary of nodes in the graph.
        _edges (set): A set of edges in the graph.
        _directed (bool): A boolean indicating whether the graph is directed.
        _weighted (bool): A boolean indicating whether the graph is weighted.
        _first_index (Literal[0, 1]): The first index of the graph.

    Methods:
        get_nodes: Returns the nodes in the graph.
        get_edges: Returns the edges in the graph.
        is_directed: Returns whether the graph is directed.
        is_weighted: Returns whether the graph is weighted.
        add_node: Adds a node to the graph.
        add_edge: Adds an edge to the graph.
        read_from_file: Reads a graph from a file.
        topological_sort: Returns a topological sort of the graph.
        is_acyclic: Returns whether the graph is acyclic.
        dijkstra: Returns the shortest path from a node to all other nodes.
        bfs: Returns a breadth-first search of the graph.
        dfs: Returns a depth-first search of the graph.
        get_edge_weight: Returns the weight of an edge.
        get_first_index: Returns the first index of the graph.
        get_node: Returns a node in the graph.
        get_edge: Returns an edge in the graph.
    """

    def __init__(
        self: Self,
        directed: bool = True,
        weighted: bool = False,
        first_index: Literal[0, 1] = 0,
    ) -> None:
        self._nodes = defaultdict(Node)
        self._edges = set()
        self._directed = directed
        self._weighted = weighted
        self._first_index = first_index

    def _is_acyclic_helper(self: Self, node: Node, visited: set, stack: set) -> bool:
        """Helper function for is_acyclic.

        Args:
            node (Node): The node.
            visited (set): A set of visited nodes.
            stack (set): A set of nodes in the current stack.

        Returns:
            bool: Whether the graph is acyclic.
        """
        visited.add(node)
        stack.add(node)
        for edge in self._edges:
            if edge.get_start_node == node:
                end_node = edge.get_end_node
                if end_node not in visited:
                    if self._is_acyclic_helper(end_node, visited, stack):
                        return True
                elif end_node in stack:
                    return True
        stack.remove(node)
        return False

    @property
    def get_nodes(self: Self) -> iter:
        """Returns the nodes in the graph.

        Returns:
            iter: The nodes in the graph.
        """
        return iter(self._nodes.values())

    @property
    def get_edges(self: Self) -> iter:
        """Returns the edges in the graph.

        Returns:
            iter: The edges in the graph.
        """
        return iter(self._edges)

    @property
    def is_directed(self: Self) -> bool:
        """Returns whether the graph is directed.

        Returns:
            bool: Whether the graph is directed.
        """
        return self._directed

    @property
    def is_weighted(self: Self) -> bool:
        """Returns whether the graph is weighted.

        Returns:
            bool: Whether the graph is weighted.
        """
        return self._weighted

    @property
    def get_first_index(self: Self) -> Literal[0, 1]:
        """Returns the first index of the graph.

        Returns:
            Literal[0, 1]: The first index of the graph.
        """
        return self._first_index

    def add_node(self: Self, value: Any = 0) -> None:
        """Adds a node to the graph.

        Args:
            value (Any): The value of the node.
        """
        index = len(self._nodes) + self._first_index
        self._nodes[index] = Node(index, value)

    def add_edge(
        self: Self, start_node_idx: int, end_node_idx: int, weight: int = 1
    ) -> None:
        """Adds an edge to the graph.

        Args:
            start_node_idx (int): The index of the start node.
            end_node_idx (int): The index of the end node.
            weight (int): The weight of the edge.

        Raises:
            KeyError: If the start or end node index is invalid.
        """
        start_node = self._nodes.get(start_node_idx)
        end_node = self._nodes.get(end_node_idx)
        if start_node is None or end_node is None:
            raise KeyError("Invalid node index.")

        self._edges.add(
            Edge(start_node, end_node, self._directed, self._weighted, weight)
        )
        if not self._directed:
            self._edges.add(
                Edge(end_node, start_node, self._directed, self._weighted, weight)
            )

    def remove_node(self: Self, node_idx: int) -> None:
        """Removes a node from the graph.

        Args:
            node_idx (int): The index of the node to remove.

        Raises:
            KeyError: If the node index is invalid.
        """
        node = self._nodes[node_idx]
        if node is None:
            raise KeyError("Invalid node index.")

        for edge in self._edges:
            if edge.get_start_node == node or edge.get_end_node == node:
                self._edges.remove(edge)
        del self._nodes[node_idx]

    def remove_edge(self: Self, start_node_idx: int, end_node_idx: int) -> None:
        """Removes an edge from the graph.

        Args:
            start_node_idx (int): The index of the start node.
            end_node_idx (int): The index of the end node.

        Raises:
            KeyError: If the start or end node index is invalid.
        """
        start_node = self._nodes.get(start_node_idx)
        end_node = self._nodes.get(end_node_idx)
        if start_node is None or end_node is None:
            raise KeyError("Invalid node index.")

        for edge in self._edges:
            if edge.get_start_node == start_node and edge.get_end_node == end_node:
                self._edges.remove(edge)
                break

        if not self._directed:
            for edge in self._edges:
                if edge.get_start_node == end_node and edge.get_end_node == start_node:
                    self._edges.remove(edge)
                    break

    def get_neighbors(self: Self, node_idx: int) -> iter:
        """Gets the neighbors of a node.

        Args:
            node_idx (int): The index of the node.

        Returns:
            iter: An iterator of the neighbors of the node.

        Raises:
            KeyError: If the node index is invalid.
        """
        node = self._nodes.get(node_idx)
        if node is None:
            raise KeyError("Invalid node index.")

        neighbors = set()
        for edge in self._edges:
            if edge.get_start_node == node:
                neighbors.add(edge.get_end_node)
        return iter(neighbors)

    def get_edge_weight(self: Self, start_node_idx: int, end_node_idx: int) -> int:
        """Gets the weight of an edge.

        Args:
            start_node_idx (int): The index of the start node.
            end_node_idx (int): The index of the end node.

        Returns:
            int: The weight of the edge.

        Raises:
            KeyError: If the start or end node index is invalid.
        """
        start_node = self._nodes.get(start_node_idx)
        end_node = self._nodes.get(end_node_idx)
        if start_node is None or end_node is None:
            raise KeyError("Invalid node index.")

        for edge in self._edges:
            if edge.get_start_node == start_node and edge.get_end_node == end_node:
                return edge.get_weight
        return 0

    def is_acyclic(self: Self) -> bool:
        """Returns whether the graph is acyclic.

        Returns:
            bool: Whether the graph is acyclic.
        """
        visited, stack = set(), set()
        for node in self._nodes.values():
            if node not in visited:
                if self._is_acyclic_helper(node, visited, stack):
                    return False
        return True

    def bfs(self: Self, start_node_idx: int) -> iter:
        """Performs a breadth-first search on the graph.

        Args:
            start_node_idx (int): The index of the start node.

        Returns:
            iter: An iterator of the visited nodes.

        Raises:
            KeyError: If the start node index is invalid.
        """
        start_node = self._nodes.get(start_node_idx)
        if start_node is None:
            raise KeyError("Invalid node index.")

        visited, queue = set(), [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(self.get_neighbors(node.get_index))
        return iter(visited)

    def dfs(self: Self, start_node_idx: int) -> iter:
        """Performs a depth-first search on the graph.

        Args:
            start_node_idx (int): The index of the start node.

        Returns:
            iter: An iterator of the visited nodes.

        Raises:
            KeyError: If the start node index is invalid.
        """
        start_node = self._nodes.get(start_node_idx)
        if start_node is None:
            raise KeyError("Invalid node index.")

        visited, stack = set(), [start_node]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self.get_neighbors(node.get_index))
        return iter(visited)

    def dijkstra(
        self: Self, start_node_idx: int, end_node_idx: int
    ) -> tuple[iter, int]:
        """Performs Dijkstra's algorithm on the graph.

        Args:
            start_node_idx (int): The index of the start node.
            end_node_idx (int): The index of the end node.

        Returns:
            tuple[iter, int]: A tuple containing an iterator of the nodes in the shortest path and the total weight of the path.

        Raises:
            ValueError: If the graph is not weighted.
            KeyError: If the start or end node index is invalid.
        """
        if not self._weighted:
            raise ValueError("Graph must be weighted.")

        start_node = self._nodes.get(start_node_idx)
        end_node = self._nodes.get(end_node_idx)
        if start_node is None or end_node is None:
            raise KeyError("Invalid node index.")

        weights = {node: float("inf") for node in self._nodes.values()}
        weights[start_node] = 0
        previous = {node: None for node in self._nodes.values()}
        priority_queue = [(0, start_node)]

        while priority_queue:
            _, node = heapq.heappop(priority_queue)
            for neighbor in self.get_neighbors(node.get_index):
                new_weight = weights[node] + self.get_edge_weight(
                    node.get_index, neighbor.get_index
                )
                if new_weight < weights[neighbor]:
                    weights[neighbor] = new_weight
                    previous[neighbor] = node
                    heapq.heappush(priority_queue, (new_weight, neighbor))

        path = []
        node = end_node
        while node is not None:
            path.append(node)
            node = previous[node]
        path.reverse()
        return iter(path), weights[end_node]

    def topological_sort(self: Self) -> iter:
        """Performs a topological sort on the graph.

        Returns:
            iter: An iterator of the nodes in the order they were visited.

        Raises:
            ValueError: If the graph is not directed or is not acyclic.
        """
        if not self._directed:
            raise ValueError("Graph must be directed.")

        if not self.is_acyclic():
            raise ValueError("Graph must be acyclic.")

        incoming_edges = {node: 0 for node in self._nodes.values()}
        for edge in self._edges:
            incoming_edges[edge.get_end_node] += 1

        queue = []
        for node in self._nodes.values():
            if incoming_edges[node] == 0:
                queue.append(node)

        sorted_nodes = []
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for neighbor in self.get_neighbors(node.get_index):
                incoming_edges[neighbor] -= 1
                if incoming_edges[neighbor] == 0:
                    queue.append(neighbor)

        return iter(sorted_nodes)

    def floyd_warshall(self: Self) -> dict[Node, dict[Node, int]]:
        """Performs Floyd-Warshall's algorithm on the graph.

        Returns:
            dict[Node, dict[Node, int]]: A dictionary containing the shortest paths between all pairs of nodes.

        Raises:
            ValueError: If the graph is not weighted.
        """
        if not self._weighted:
            raise ValueError("Graph must be weighted.")

        distances = {
            node: {node: float("inf") for node in self._nodes.values()}
            for node in self._nodes.values()
        }
        for node in self._nodes.values():
            distances[node][node] = 0

        for edge in self._edges:
            distances[edge.get_start_node][edge.get_end_node] = edge.get_weight

        for node in self._nodes.values():
            for start_node in self._nodes.values():
                for end_node in self._nodes.values():
                    new_distance = (
                        distances[start_node][node] + distances[node][end_node]
                    )
                    if new_distance < distances[start_node][end_node]:
                        distances[start_node][end_node] = new_distance

        return distances

    def kruskal(self: Self, minimum: bool = True) -> iter:
        """Performs Kruskal's algorithm on the graph.

        Args:
            minimum (bool, optional): Whether to find the minimum or maximum spanning tree. Defaults to True.

        Returns:
            iter: An iterator of the edges in the minimum or maximum spanning tree.

        Raises:
            ValueError: If the graph is not weighted.
        """
        if not self._weighted:
            raise ValueError("Graph must be weighted.")

        sorted_edges = sorted(
            self._edges, key=lambda edge: edge.get_weight, reverse=not minimum
        )
        disjoint_set = DisjointSet()
        for node in self._nodes.values():
            disjoint_set.make_set(node)

        spanning_tree = []
        for edge in sorted_edges:
            start_node = edge.get_start_node
            end_node = edge.get_end_node
            if disjoint_set.find_set(start_node) != disjoint_set.find_set(end_node):
                spanning_tree.append(edge)
                disjoint_set.union(start_node, end_node)

        return iter(spanning_tree)

    def read_from_file(self: Self, file_name: str) -> None:
        """Reads a graph from a file.

        Args:
            file_name (str): The name of the file to read from.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is in an invalid format.
        """
        with open(file_name, "r") as file:
            try:
                for index, line in enumerate(file):
                    if index == 0:
                        for _ in range(int(line)):
                            self.add_node()
                    else:
                        if self._weighted:
                            start_node_idx, end_node_idx, weight = map(
                                int, line.split(" ")
                            )
                            self.add_edge(start_node_idx, end_node_idx, weight)
                        else:
                            start_node_idx, end_node_idx = map(int, line.split(" "))
                            self.add_edge(start_node_idx, end_node_idx)
            except ValueError:
                raise ValueError("Invalid file format.")

    def __str__(self) -> str:
        """Returns a string representation of the graph.

        Returns:
            str: A string representation of the graph.
        """
        return_str = ""
        for index, edge in enumerate(self._edges):
            if index == len(self._edges) - 1:
                return_str += str(edge)
            else:
                return_str += str(edge) + "\n"
        return return_str
