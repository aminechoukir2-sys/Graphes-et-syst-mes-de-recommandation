import numpy as np

from src.config import ProjectConfig
from src.interfaces import BaseRecommender


class PageRankRecommender(BaseRecommender):
    """
    PageRank from scratch sur sous-graphe local.
    """

    def __init__(self, graph, candidate_provider, config: ProjectConfig):
        self.graph = graph
        self.candidate_provider = candidate_provider
        self.config = config

    @property
    def name(self):
        return f"pagerank_{self.candidate_provider.name}"

    def score_items(self, user_id):
        candidate_list = self.candidate_provider.find_candidate_movies(user_id)
        candidates = set(candidate_list)

        if not candidates:
            return {}

        local_nodes = self.candidate_provider.local_nodes(user_id)

        if not local_nodes:
            return {}

        nodes = list(local_nodes)
        user_node = self.graph.user_node(user_id)

        rank = {node: 1 / len(nodes) for node in nodes}
        personalization = {node: 0 for node in nodes}

        if user_node in personalization:
            personalization[user_node] = 1

        local_neighbors = {}

        for node in nodes:
            local_neighbors[node] = [
                (neighbor, weight)
                for neighbor, weight in self.graph.get_neighbors(node)
                if neighbor in local_nodes
            ]

        for _ in range(self.config.pagerank_max_iter):
            new_rank = {
                node: (1 - self.config.pagerank_damping) * personalization[node]
                for node in nodes
            }

            for node in nodes:
                neighbors = local_neighbors[node]

                if not neighbors:
                    continue

                total_weight = sum(weight for _, weight in neighbors)

                if total_weight == 0:
                    continue

                for neighbor, weight in neighbors:
                    new_rank[neighbor] += (
                        self.config.pagerank_damping
                        * rank[node]
                        * (weight / total_weight)
                    )

            diff = sum(abs(new_rank[node] - rank[node]) for node in nodes)
            rank = new_rank

            if diff < self.config.pagerank_tolerance:
                break

        scores = {}

        for index, movie_id in enumerate(candidate_list):
            movie_node = self.graph.movie_node(movie_id)

            # Si le nœud est dans le PageRank, on prend le score.
            # Sinon on garde un petit score de secours basé sur le rang candidat.
            scores[movie_id] = rank.get(movie_node, 1e-9 / (index + 1))

        return scores


class RandomWalkRecommender(BaseRecommender):
    """
    Random Walk from scratch.
    """

    def __init__(self, graph, candidate_provider, config: ProjectConfig):
        self.graph = graph
        self.candidate_provider = candidate_provider
        self.config = config

    @property
    def name(self):
        return f"random_walk_{self.candidate_provider.name}"

    def score_items(self, user_id):
        candidate_list = self.candidate_provider.find_candidate_movies(user_id)
        candidates = set(candidate_list)

        if not candidates:
            return {}

        try:
            user_number = int(user_id.replace("U", ""))
        except ValueError:
            user_number = 0

        rng = np.random.default_rng(self.config.random_state + user_number)
        start_node = self.graph.user_node(user_id)

        if start_node not in self.graph.adjacency:
            return {}

        current_node = start_node
        visits = {}

        for _ in range(self.config.random_walk_steps):
            if rng.random() < self.config.random_walk_restart_probability:
                current_node = start_node
                continue

            neighbors = self.graph.get_neighbors(current_node)

            if not neighbors:
                current_node = start_node
                continue

            neighbor_nodes = [node for node, _ in neighbors]
            weights = np.array([weight for _, weight in neighbors], dtype=float)

            if weights.sum() == 0:
                current_node = start_node
                continue

            probabilities = weights / weights.sum()
            current_node = rng.choice(neighbor_nodes, p=probabilities)

            if current_node.startswith("movie::"):
                movie_id = self.graph.clean_movie_id(current_node)

                if movie_id in candidates:
                    visits[movie_id] = visits.get(movie_id, 0) + 1

        total = sum(visits.values())

        if total == 0:
            # Correction V6 :
            # si la marche aléatoire n'a pas visité de candidat,
            # on ne retourne pas des scores tous à 0.
            # On utilise le rang des candidats comme score de secours.
            return {
                movie_id: 1 / (index + 1)
                for index, movie_id in enumerate(candidate_list)
            }

        scores = {}

        for index, movie_id in enumerate(candidate_list):
            # Score principal : fréquence de visite.
            # Petit bonus stable : rang du candidat, pour éviter les égalités massives à 0.
            scores[movie_id] = visits.get(movie_id, 0) / total + 1e-9 / (index + 1)

        return scores
