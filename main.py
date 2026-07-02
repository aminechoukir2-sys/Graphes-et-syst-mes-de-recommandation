import argparse

from src.complexity import ComplexityAnalyzer
from src.config import ProjectConfig
from src.services import RecommendationProjectService


ALL_METHODS = [
    "pagerank_alone",
    "random_walk_alone",
    "pagerank_bfs",
    "pagerank_dfs",
    "random_walk_bfs",
    "random_walk_dfs",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--movies", type=int, default=200)
    parser.add_argument("--ratings", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eval-users", type=int, default=50)

    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-visited", type=int, default=300)
    parser.add_argument("--max-candidates", type=int, default=100)

    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        choices=ALL_METHODS,
    )

    parser.add_argument("--force-regenerate", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = ProjectConfig(
        n_users=args.users,
        n_movies=args.movies,
        n_ratings=args.ratings,
        top_k=args.top_k,
        max_eval_users=args.eval_users,
        traversal_max_depth=args.max_depth,
        traversal_max_visited_nodes=args.max_visited,
        max_candidate_movies=args.max_candidates,
    )

    print("===== PROJET M1 DATA : GRAPHES ET RECOMMANDATION =====")
    print("\nConfiguration")
    print(config)

    service = RecommendationProjectService(
        config=config,
        force_regenerate=args.force_regenerate,
    )

    project = service.prepare()

    movies = project["movies"]
    ratings = project["ratings"]
    train = project["train"]
    test = project["test"]
    graph = project["graph"]
    providers = project["providers"]
    recommenders = project["recommenders"]

    print("\nDataset")
    print("Utilisateurs demandés :", config.n_users)
    print("Films demandés :", config.n_movies)
    print("Notes générées :", len(ratings))
    print("Train :", len(train))
    print("Test :", len(test))
    print("Utilisateurs présents :", ratings["user_id"].nunique())
    print("Films présents :", ratings["movie_id"].nunique())

    print("\nGraphe")
    print("Nombre de noeuds :", len(graph.nodes()))
    print("Nombre d'arêtes train :", len(train))

    print("\nComplexité théorique")
    print(ComplexityAnalyzer().summary())

    print("\nÉvaluation BFS/DFS comme générateurs de candidats")
    candidate_results = project["candidate_evaluator"].compare(
        [
            providers["bfs"],
            providers["dfs"],
        ]
    )
    print(candidate_results)

    print("\nÉvaluation des méthodes de recommandation")
    selected_recommenders = [recommenders[name] for name in args.methods]
    recommendation_results = project["recommendation_evaluator"].compare(selected_recommenders)
    print(recommendation_results)

    print("\nMéthodes comparées :")
    for method in args.methods:
        print("-", method)

    print("\nLecture :")
    print("- Precision@K : qualité du top K")
    print("- Recall@K : capacité à retrouver les bons films cachés")
    print("- F1@K : équilibre précision/rappel")
    print("- avg_time_ms : temps moyen par utilisateur")
    print("- candidate_recall : qualité de BFS/DFS pour retrouver des candidats pertinents")


if __name__ == "__main__":
    main()
