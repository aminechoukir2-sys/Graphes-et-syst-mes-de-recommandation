from collections import defaultdict

from src.config import ProjectConfig
from src.interfaces import BaseCandidateProvider


class UserHistory:
    def __init__(self, train):
        self.train = train

    def seen_movies(self, user_id):
        return set(self.train.loc[self.train["user_id"] == user_id, "movie_id"])


class CandidateExtractor:
    """
    Transforme un parcours de graphe en films candidats.

    Correction V6 :
    on ne prend pas seulement les films visibles directement dans l'ordre BFS/DFS.
    On regarde aussi les films notés par les utilisateurs atteints par le parcours.
    Cela évite que BFS s'arrête trop tôt et retourne de mauvais candidats.
    """

    def __init__(self, graph, train, config: ProjectConfig):
        self.graph = graph
        self.config = config
        self.history = UserHistory(train)

    def from_nodes(self, user_id, traversal_nodes_with_depth):
        seen = self.history.seen_movies(user_id)
        start_user_node = self.graph.user_node(user_id)

        candidate_scores = defaultdict(float)

        for node, depth in traversal_nodes_with_depth:
            # Cas 1 : le parcours a directement atteint un film non vu.
            if node.startswith("movie::"):
                movie_id = self.graph.clean_movie_id(node)

                if movie_id not in seen:
                    candidate_scores[movie_id] += 1.0 / (depth + 1)

            # Cas 2 : le parcours atteint un autre utilisateur.
            # On récupère alors les films vus par cet utilisateur.
            if node.startswith("user::") and node != start_user_node:
                for neighbor, weight in self.graph.get_neighbors(node):
                    if not neighbor.startswith("movie::"):
                        continue

                    movie_id = self.graph.clean_movie_id(neighbor)

                    if movie_id in seen:
                        continue

                    # Plus le chemin est court et plus la note est haute,
                    # plus le candidat est prioritaire.
                    candidate_scores[movie_id] += weight / (depth + 1)

        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            movie_id
            for movie_id, _ in sorted_candidates[: self.config.max_candidate_movies]
        ]

    def nodes_from_candidates(self, candidate_movies):
        return {
            self.graph.movie_node(movie_id)
            for movie_id in candidate_movies
        }


class DirectCandidateProvider(BaseCandidateProvider):
    """
    Version sans BFS/DFS.

    On récupère des candidats à partir des films vus par des utilisateurs proches.
    """

    def __init__(self, graph, train, config: ProjectConfig):
        self.graph = graph
        self.train = train
        self.config = config
        self.history = UserHistory(train)

    @property
    def name(self):
        return "alone"

    def local_nodes(self, user_id):
        user_node = self.graph.user_node(user_id)
        seen_movies = self.history.seen_movies(user_id)

        nodes = {user_node}

        for movie_id in seen_movies:
            movie_node = self.graph.movie_node(movie_id)
            nodes.add(movie_node)

            for other_user_node, _ in self.graph.get_neighbors(movie_node):
                nodes.add(other_user_node)

                for candidate_movie_node, _ in self.graph.get_neighbors(other_user_node):
                    nodes.add(candidate_movie_node)

                    if len(nodes) >= self.config.traversal_max_visited_nodes:
                        return nodes

        return nodes

    def find_candidate_movies(self, user_id):
        seen = self.history.seen_movies(user_id)
        candidate_scores = defaultdict(float)

        for node in self.local_nodes(user_id):
            if not node.startswith("movie::"):
                continue

            movie_id = self.graph.clean_movie_id(node)

            if movie_id in seen:
                continue

            candidate_scores[movie_id] += 1.0

        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            movie_id
            for movie_id, _ in sorted_candidates[: self.config.max_candidate_movies]
        ]


class BFSCandidateProvider(BaseCandidateProvider):
    def __init__(self, graph, train, config: ProjectConfig):
        self.graph = graph
        self.config = config
        self.extractor = CandidateExtractor(graph, train, config)

    @property
    def name(self):
        return "bfs"

    def traversal_order(self, user_id):
        start = self.graph.user_node(user_id)

        return self.graph.bfs_with_depth(
            start,
            self.config.traversal_max_depth,
            self.config.traversal_max_visited_nodes,
        )

    def local_nodes(self, user_id):
        traversal = self.traversal_order(user_id)
        nodes = {node for node, _ in traversal}

        candidates = self.extractor.from_nodes(user_id, traversal)
        nodes.update(self.extractor.nodes_from_candidates(candidates))

        return nodes

    def find_candidate_movies(self, user_id):
        traversal = self.traversal_order(user_id)
        return self.extractor.from_nodes(user_id, traversal)


class DFSCandidateProvider(BaseCandidateProvider):
    def __init__(self, graph, train, config: ProjectConfig):
        self.graph = graph
        self.config = config
        self.extractor = CandidateExtractor(graph, train, config)

    @property
    def name(self):
        return "dfs"

    def traversal_order(self, user_id):
        start = self.graph.user_node(user_id)

        return self.graph.dfs_with_depth(
            start,
            self.config.traversal_max_depth,
            self.config.traversal_max_visited_nodes,
        )

    def local_nodes(self, user_id):
        traversal = self.traversal_order(user_id)
        nodes = {node for node, _ in traversal}

        candidates = self.extractor.from_nodes(user_id, traversal)
        nodes.update(self.extractor.nodes_from_candidates(candidates))

        return nodes

    def find_candidate_movies(self, user_id):
        traversal = self.traversal_order(user_id)
        return self.extractor.from_nodes(user_id, traversal)
