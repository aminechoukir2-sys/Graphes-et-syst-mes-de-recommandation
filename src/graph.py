from collections import defaultdict, deque

from src.interfaces import BaseGraph


class BipartiteGraph(BaseGraph):
    """
    Graphe bipartite utilisateurs × films avec liste d'adjacence.
    """

    def __init__(self):
        self.adjacency = defaultdict(list)

    @staticmethod
    def user_node(user_id):
        return f"user::{user_id}"

    @staticmethod
    def movie_node(movie_id):
        return f"movie::{movie_id}"

    @staticmethod
    def clean_movie_id(node):
        return node.replace("movie::", "")

    def add_edge(self, user_id, movie_id, weight):
        user = self.user_node(user_id)
        movie = self.movie_node(movie_id)

        self.adjacency[user].append((movie, weight))
        self.adjacency[movie].append((user, weight))

    def build_from_ratings(self, ratings):
        for row in ratings.itertuples(index=False):
            self.add_edge(row.user_id, row.movie_id, row.rating / 5)

    def get_neighbors(self, node):
        return self.adjacency.get(node, [])

    def nodes(self):
        return list(self.adjacency.keys())

    def bfs_with_depth(self, start_node, max_depth, max_visited_nodes):
        visited = set()
        queue = deque([(start_node, 0)])
        order = []

        while queue and len(order) < max_visited_nodes:
            node, depth = queue.popleft()

            if node in visited:
                continue

            visited.add(node)
            order.append((node, depth))

            if depth >= max_depth:
                continue

            for neighbor, _ in self.get_neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return order

    def dfs_with_depth(self, start_node, max_depth, max_visited_nodes):
        visited = set()
        stack = [(start_node, 0)]
        order = []

        while stack and len(order) < max_visited_nodes:
            node, depth = stack.pop()

            if node in visited:
                continue

            visited.add(node)
            order.append((node, depth))

            if depth >= max_depth:
                continue

            for neighbor, _ in reversed(self.get_neighbors(node)):
                if neighbor not in visited:
                    stack.append((neighbor, depth + 1))

        return order


class GraphBuilder:
    def build(self, train):
        graph = BipartiteGraph()
        graph.build_from_ratings(train)
        return graph
