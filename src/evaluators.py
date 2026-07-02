import time

import pandas as pd

from src.config import ProjectConfig
from src.metrics import F1AtK, PrecisionAtK, RecallAtK


class RelevantItemsProvider:
    """
    Films pertinents = films du test avec note >= 4.
    """

    def __init__(self, test, min_rating=4):
        self.test = test
        self.min_rating = min_rating

    def get(self, user_id):
        rows = self.test[
            (self.test["user_id"] == user_id)
            & (self.test["rating"] >= self.min_rating)
        ]

        return set(rows["movie_id"])


class CandidateEvaluator:
    """
    Évalue BFS/DFS comme générateurs de candidats.
    """

    def __init__(self, test, config: ProjectConfig):
        self.test = test
        self.config = config
        self.relevant = RelevantItemsProvider(test)

    def evaluate(self, provider):
        users = sorted(self.test["user_id"].unique())

        if self.config.max_eval_users is not None:
            users = users[: self.config.max_eval_users]

        recalls = []
        counts = []
        times = []

        for user_id in users:
            relevant_items = self.relevant.get(user_id)

            if not relevant_items:
                continue

            start = time.perf_counter()
            candidates = set(provider.find_candidate_movies(user_id))
            end = time.perf_counter()

            recalls.append(len(candidates & relevant_items) / len(relevant_items))
            counts.append(len(candidates))
            times.append((end - start) * 1000)

        if not recalls:
            return {
                "provider": provider.name,
                "candidate_recall": 0,
                "avg_candidates": 0,
                "avg_time_ms": 0,
                "complexity": "O(V + E) worst-case, limited by depth/nodes",
                "n_users_evaluated": 0,
            }

        return {
            "provider": provider.name,
            "candidate_recall": sum(recalls) / len(recalls),
            "avg_candidates": sum(counts) / len(counts),
            "avg_time_ms": sum(times) / len(times),
            "complexity": "O(V + E) worst-case, limited by depth/nodes",
            "n_users_evaluated": len(recalls),
        }

    def compare(self, providers):
        return pd.DataFrame([self.evaluate(provider) for provider in providers])


class RecommendationEvaluator:
    """
    Évalue les recommandations finales.
    """

    def __init__(self, test, config: ProjectConfig):
        self.test = test
        self.config = config
        self.relevant = RelevantItemsProvider(test)
        self.precision = PrecisionAtK()
        self.recall = RecallAtK()
        self.f1 = F1AtK()

    def evaluate(self, recommender):
        users = sorted(self.test["user_id"].unique())

        if self.config.max_eval_users is not None:
            users = users[: self.config.max_eval_users]

        precisions = []
        recalls = []
        f1_scores = []
        times = []

        for user_id in users:
            relevant_items = self.relevant.get(user_id)

            if not relevant_items:
                continue

            start = time.perf_counter()
            recos = recommender.recommend(user_id, self.config.top_k)
            end = time.perf_counter()

            recommended_items = [movie_id for movie_id, _ in recos]

            p = self.precision.calculate(
                recommended_items,
                relevant_items,
                self.config.top_k,
            )

            r = self.recall.calculate(
                recommended_items,
                relevant_items,
                self.config.top_k,
            )

            f = self.f1.calculate(
                recommended_items,
                relevant_items,
                self.config.top_k,
            )

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f)
            times.append((end - start) * 1000)

        if not precisions:
            return {
                "method": recommender.name,
                "precision_at_k": 0,
                "recall_at_k": 0,
                "f1_at_k": 0,
                "avg_time_ms": 0,
                "n_users_evaluated": 0,
            }

        return {
            "method": recommender.name,
            "precision_at_k": sum(precisions) / len(precisions),
            "recall_at_k": sum(recalls) / len(recalls),
            "f1_at_k": sum(f1_scores) / len(f1_scores),
            "avg_time_ms": sum(times) / len(times),
            "n_users_evaluated": len(precisions),
        }

    def compare(self, recommenders):
        return pd.DataFrame([self.evaluate(recommender) for recommender in recommenders])
