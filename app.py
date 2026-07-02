import pandas as pd
import streamlit as st

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


METHOD_COMPLEXITIES = {
    "pagerank_alone": "O(T × E_local)",
    "random_walk_alone": "O(S)",
    "pagerank_bfs": "O(BFS) + O(T × E_candidates)",
    "pagerank_dfs": "O(DFS) + O(T × E_candidates)",
    "random_walk_bfs": "O(BFS) + O(S)",
    "random_walk_dfs": "O(DFS) + O(S)",
}


METHOD_EXPLANATIONS = {
    "pagerank_alone": "PageRank seul : il utilise directement le graphe sans filtre BFS/DFS.",
    "random_walk_alone": "Random Walk seul : il marche dans le graphe sans filtre BFS/DFS.",
    "pagerank_bfs": "BFS trouve les films candidats, puis PageRank calcule le score.",
    "pagerank_dfs": "DFS trouve les films candidats, puis PageRank calcule le score.",
    "random_walk_bfs": "BFS trouve les films candidats, puis Random Walk calcule le score.",
    "random_walk_dfs": "DFS trouve les films candidats, puis Random Walk calcule le score.",
}


st.set_page_config(
    page_title="Graphes et systèmes de recommandation",
    layout="wide",
)


@st.cache_resource
def load_project(config_tuple, force_regenerate):
    config = ProjectConfig(
        n_users=config_tuple[0],
        n_movies=config_tuple[1],
        n_ratings=config_tuple[2],
        top_k=config_tuple[3],
        max_eval_users=config_tuple[4],
        traversal_max_depth=config_tuple[5],
        traversal_max_visited_nodes=config_tuple[6],
        max_candidate_movies=config_tuple[7],
    )

    service = RecommendationProjectService(
        config=config,
        force_regenerate=force_regenerate,
    )

    return service.prepare(), config


@st.cache_data(show_spinner=False)
def evaluate_all_methods(_project, selected_method_names, top_k):
    """
    Évalue toutes les méthodes ou seulement les méthodes demandées.
    _project commence par _ pour que Streamlit ne tente pas de hasher l'objet.
    """
    recommenders = _project["recommenders"]
    evaluator = _project["recommendation_evaluator"]

    selected_recommenders = [
        recommenders[name]
        for name in selected_method_names
    ]

    return evaluator.compare(selected_recommenders)


def enrich_recommendations(recommendations, movies):
    rows = []

    for rank, (movie_id, score) in enumerate(recommendations, start=1):
        movie = movies[movies["movie_id"] == movie_id]

        if movie.empty:
            continue

        movie = movie.iloc[0]

        rows.append(
            {
                "rank": rank,
                "movie_id": movie_id,
                "title": movie["title"],
                "genre": movie["genre"],
                "year": movie["year"],
                "score": round(score, 6),
            }
        )

    return pd.DataFrame(rows)


def format_comparison_results(results, top_k):
    """
    Renomme les colonnes pour afficher clairement Precision@K, Recall@K,
    F1@K, temps d'exécution et complexité.
    """
    if results.empty:
        return results

    results = results.copy()

    results["Complexité"] = results["method"].map(METHOD_COMPLEXITIES)
    results["Explication"] = results["method"].map(METHOD_EXPLANATIONS)

    results = results.rename(
        columns={
            "method": "Algorithme",
            "precision_at_k": f"Precision@{top_k}",
            "recall_at_k": f"Recall@{top_k}",
            "f1_at_k": f"F1@{top_k}",
            "avg_time_ms": "Temps execution moyen (ms)",
            "n_users_evaluated": "Utilisateurs évalués",
        }
    )

    ordered_columns = [
        "Algorithme",
        f"Precision@{top_k}",
        f"Recall@{top_k}",
        f"F1@{top_k}",
        "Temps execution moyen (ms)",
        "Complexité",
        "Utilisateurs évalués",
        "Explication",
    ]

    existing_columns = [col for col in ordered_columns if col in results.columns]

    return results[existing_columns]


def metrics_chart_data(formatted_results, top_k):
    metric_columns = [
        f"Precision@{top_k}",
        f"Recall@{top_k}",
        f"F1@{top_k}",
    ]

    available_columns = [
        col for col in metric_columns
        if col in formatted_results.columns
    ]

    return formatted_results.set_index("Algorithme")[available_columns]


def time_chart_data(formatted_results):
    return formatted_results.set_index("Algorithme")[["Temps execution moyen (ms)"]]


def main():
    st.title("Graphes et systèmes de recommandation")
    st.write(
        "Version finale : calcul automatique pour tous les algorithmes + comparaison graphique de 3 algorithmes."
    )

    with st.sidebar:
        st.header("Configuration dataset")

        n_users = st.number_input(
            "Nombre d'utilisateurs",
            min_value=10,
            max_value=50000,
            value=500,
            step=100,
        )

        n_movies = st.number_input(
            "Nombre de films",
            min_value=10,
            max_value=20000,
            value=200,
            step=100,
        )

        n_ratings = st.number_input(
            "Nombre de notes",
            min_value=100,
            max_value=10000000,
            value=10000,
            step=1000,
        )

        st.header("Évaluation")

        top_k = st.number_input(
            "K pour Precision@K / Recall@K",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )

        eval_users = st.number_input(
            "Nombre d'utilisateurs évalués",
            min_value=1,
            max_value=1000,
            value=50,
            step=10,
        )

        st.header("BFS / DFS")

        max_depth = st.number_input(
            "Profondeur max",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        max_visited = st.number_input(
            "Nombre max de nœuds visités",
            min_value=20,
            max_value=5000,
            value=300,
            step=50,
        )

        max_candidates = st.number_input(
            "Nombre max de films candidats",
            min_value=5,
            max_value=1000,
            value=100,
            step=10,
        )

        force_regenerate = st.checkbox(
            "Forcer la régénération du dataset",
            value=False,
        )

        st.header("Choix des algorithmes")

        selected_methods = st.multiselect(
            "Choisis exactement trois algorithmes pour les graphes",
            ALL_METHODS,
            default=[
                "pagerank_alone",
                "random_walk_alone",
                "pagerank_bfs",
            ],
        )

    config_tuple = (
        int(n_users),
        int(n_movies),
        int(n_ratings),
        int(top_k),
        int(eval_users),
        int(max_depth),
        int(max_visited),
        int(max_candidates),
    )

    project, config = load_project(config_tuple, force_regenerate)

    movies = project["movies"]
    ratings = project["ratings"]
    train = project["train"]
    test = project["test"]
    providers = project["providers"]
    recommenders = project["recommenders"]

    tab_data, tab_bfs_dfs, tab_reco, tab_compare, tab_complexity = st.tabs(
        [
            "Dataset",
            "Rôle BFS / DFS",
            "Recommandations",
            "Comparaison finale",
            "Complexité",
        ]
    )

    with tab_data:
        st.subheader("Dataset généré")

        st.write(
            {
                "utilisateurs_demandés": config.n_users,
                "films_demandés": config.n_movies,
                "notes": len(ratings),
                "train": len(train),
                "test": len(test),
                "utilisateurs_présents": ratings["user_id"].nunique(),
                "films_présents": ratings["movie_id"].nunique(),
            }
        )

        st.write("Aperçu des films")
        st.dataframe(movies.head(20), use_container_width=True)

        st.write("Aperçu des notes")
        st.dataframe(ratings.head(20), use_container_width=True)

        st.write("Aperçu train")
        st.dataframe(train.head(20), use_container_width=True)

        st.write("Aperçu test")
        st.dataframe(test.head(20), use_container_width=True)

    with tab_bfs_dfs:
        st.subheader("BFS et DFS comme générateurs de candidats")

        user_ids = sorted(train["user_id"].unique())

        selected_user = st.selectbox(
            "Choisir un utilisateur",
            user_ids,
            key="bfs_dfs_user",
        )

        for provider_name in ["bfs", "dfs"]:
            provider = providers[provider_name]
            candidates = provider.find_candidate_movies(selected_user)

            st.write(f"### {provider_name.upper()}")
            st.write("Rôle : explorer le graphe et proposer des films candidats.")
            st.write("Complexité théorique : O(V + E) au pire.")
            st.write("Dans le code, on limite avec : profondeur max, nœuds visités max, films candidats max.")
            st.write("Nombre de films candidats :", len(candidates))
            st.write(candidates[:30])

        st.write("### Tableau BFS / DFS")

        candidate_results = project["candidate_evaluator"].compare(
            [
                providers["bfs"],
                providers["dfs"],
            ]
        )

        st.dataframe(candidate_results, use_container_width=True)

    with tab_reco:
        st.subheader("Films recommandés par utilisateur")

        user_ids = sorted(train["user_id"].unique())

        selected_user = st.selectbox(
            "Choisir un utilisateur",
            user_ids,
            key="reco_user",
        )

        user_ratings = train[train["user_id"] == selected_user].merge(
            movies,
            on="movie_id",
            how="left",
        )

        st.write("Films déjà notés par l'utilisateur")
        st.dataframe(
            user_ratings[
                ["user_id", "movie_id", "title", "genre", "year", "rating"]
            ].sort_values("rating", ascending=False),
            use_container_width=True,
        )

        st.write("### Recommandations des six algorithmes")

        for method in ALL_METHODS:
            recommender = recommenders[method]
            recommendations = recommender.recommend(
                selected_user,
                config.top_k,
            )

            st.write(f"### {method}")
            st.caption(METHOD_EXPLANATIONS[method])
            st.write("Complexité :", METHOD_COMPLEXITIES[method])

            st.dataframe(
                enrich_recommendations(recommendations, movies),
                use_container_width=True,
            )

    with tab_compare:
        st.subheader("Comparaison finale")

        st.write("### 1. Tableau complet : les six algorithmes")

        all_raw_results = project["recommendation_evaluator"].compare(
            [recommenders[name] for name in ALL_METHODS]
        )

        all_results = format_comparison_results(all_raw_results, config.top_k)

        st.dataframe(all_results, use_container_width=True)

        st.info(
            "Ce tableau calcule Precision@K, Recall@K, F1@K, temps d'exécution et complexité pour les 6 algorithmes."
        )

        st.write("### 2. Graphes pour trois algorithmes choisis")

        if len(selected_methods) != 3:
            st.warning("Choisis exactement trois algorithmes dans la barre latérale pour afficher les graphes comparatifs.")
        else:
            selected_results = all_results[
                all_results["Algorithme"].isin(selected_methods)
            ].copy()

            st.write(
                f"Comparaison graphique entre : **{selected_methods[0]}**, **{selected_methods[1]}**, **{selected_methods[2]}**"
            )

            st.write("### Tableau des trois algorithmes sélectionnés")
            st.dataframe(selected_results, use_container_width=True)

            st.write("### Graphique en barres Precision@K / Recall@K / F1@K")
            selected_metrics_df = metrics_chart_data(selected_results, config.top_k)
            st.bar_chart(selected_metrics_df)

            st.write("### Graphe linéaire Precision / Recall / F1")
            line_df = selected_metrics_df.T
            line_df.index.name = "Métrique"
            st.line_chart(line_df)

            st.write("### Graphique du temps d'exécution")
            st.bar_chart(time_chart_data(selected_results))

            st.write("### Interprétation rapide")

            best_precision = selected_results.sort_values(
                by=f"Precision@{config.top_k}",
                ascending=False,
            ).iloc[0]["Algorithme"]

            best_recall = selected_results.sort_values(
                by=f"Recall@{config.top_k}",
                ascending=False,
            ).iloc[0]["Algorithme"]

            fastest = selected_results.sort_values(
                by="Temps execution moyen (ms)",
                ascending=True,
            ).iloc[0]["Algorithme"]

            st.write(
                {
                    f"Meilleur Precision@{config.top_k}": best_precision,
                    f"Meilleur Recall@{config.top_k}": best_recall,
                    "Plus rapide": fastest,
                }
            )

            st.write("### Films recommandés par les trois algorithmes")

            user_ids = sorted(train["user_id"].unique())

            comparison_user = st.selectbox(
                "Utilisateur pour afficher les recommandations",
                user_ids,
                key="comparison_user",
            )

            col1, col2, col3 = st.columns(3)

            for column, method in zip([col1, col2, col3], selected_methods):
                with column:
                    recommender = recommenders[method]
                    recommendations = recommender.recommend(
                        comparison_user,
                        config.top_k,
                    )

                    st.write(f"#### {method}")
                    st.caption(METHOD_EXPLANATIONS[method])
                    st.write("Complexité :", METHOD_COMPLEXITIES[method])

                    st.dataframe(
                        enrich_recommendations(recommendations, movies),
                        use_container_width=True,
                    )

        st.write("### Métriques utilisées")

        st.markdown(
            f"""
            - **Precision@{config.top_k}** : parmi les {config.top_k} films recommandés, combien sont pertinents.
            - **Recall@{config.top_k}** : parmi les bons films cachés dans le test, combien sont retrouvés.
            - **F1@{config.top_k}** : équilibre entre Precision et Recall.
            - **Temps execution moyen (ms)** : temps moyen par utilisateur.
            - **Complexité** : coût théorique de l'algorithme.
            """
        )

    with tab_complexity:
        st.subheader("Complexité théorique")

        complexity_df = ComplexityAnalyzer().summary()
        st.dataframe(complexity_df, use_container_width=True)

        st.markdown(
            """
            ### Résumé des complexités

            | Méthode | Complexité |
            |---|---|
            | Construction du graphe | O(E) |
            | BFS | O(V + E) au pire |
            | DFS | O(V + E) au pire |
            | PageRank | O(T × E_local) |
            | Random Walk | O(S) |
            | Precision@K / Recall@K | O(K) par utilisateur |

            Avec :

            - **V** = nombre de nœuds
            - **E** = nombre d'arêtes / notes
            - **T** = nombre d'itérations PageRank
            - **S** = nombre de pas Random Walk
            - **K** = nombre de recommandations

            Dans ce projet, BFS et DFS sont limités par profondeur et nombre maximum de nœuds visités.
            Cela évite que l'exécution devienne trop lente quand le dataset augmente.
            """
        )


if __name__ == "__main__":
    main()
