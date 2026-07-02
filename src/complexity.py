import pandas as pd


class ComplexityAnalyzer:
    """
    Tableau de complexité théorique.
    """

    def summary(self):
        rows = [
            {
                "method": "Graph construction",
                "time_complexity": "O(E)",
                "memory_complexity": "O(V + E)",
                "role": "Construire la liste d'adjacence",
            },
            {
                "method": "BFS",
                "time_complexity": "O(V + E) worst-case",
                "memory_complexity": "O(V)",
                "role": "Explorer par niveaux et proposer des candidats",
            },
            {
                "method": "DFS",
                "time_complexity": "O(V + E) worst-case",
                "memory_complexity": "O(V)",
                "role": "Explorer en profondeur et proposer des candidats",
            },
            {
                "method": "PageRank",
                "time_complexity": "O(T × E_local)",
                "memory_complexity": "O(V_local)",
                "role": "Scorer les films par propagation de score",
            },
            {
                "method": "Random Walk",
                "time_complexity": "O(S)",
                "memory_complexity": "O(films visités)",
                "role": "Scorer les films par fréquence de visite",
            },
            {
                "method": "Precision@K / Recall@K",
                "time_complexity": "O(K) par utilisateur",
                "memory_complexity": "O(K)",
                "role": "Évaluer les recommandations",
            },
        ]

        return pd.DataFrame(rows)
