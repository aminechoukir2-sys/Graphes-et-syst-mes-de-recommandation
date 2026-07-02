from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectConfig:
    n_users: int = 500
    n_movies: int = 200

    # 500000 notes par défaut pour éviter Precision/Recall = 0 sur un dataset trop sparse.
    # Tu peux augmenter ou diminuer cette valeur dans Streamlit ou en ligne de commande.
    n_ratings: int = 10000

    test_ratio: float = 0.20
    random_state: int = 42

    top_k: int = 5
    max_eval_users: Optional[int] = 50

    traversal_max_depth: int = 3
    traversal_max_visited_nodes: int = 300
    max_candidate_movies: int = 100

    pagerank_damping: float = 0.85
    pagerank_max_iter: int = 20
    pagerank_tolerance: float = 1e-6

    random_walk_steps: int = 1500
    random_walk_restart_probability: float = 0.15

    data_dir: str = "data"
