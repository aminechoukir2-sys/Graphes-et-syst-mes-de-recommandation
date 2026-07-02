import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.interfaces import BaseDatasetGenerator


class SyntheticMovieLensDatasetGenerator(BaseDatasetGenerator):
    """
    Génère un dataset sparse inspiré de MovieLens.

    Version stable :
    - génère exactement n_ratings notes si n_ratings <= users × movies ;
    - crée des préférences utilisateurs cohérentes ;
    - cache des films bien notés dans test pour obtenir des Precision@K / Recall@K utiles.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.rng = np.random.default_rng(config.random_state)

    def _validate(self):
        max_pairs = self.config.n_users * self.config.n_movies

        if self.config.n_ratings > max_pairs:
            raise ValueError(
                f"n_ratings={self.config.n_ratings} dépasse users × movies={max_pairs}."
            )

        if self.config.n_users <= 0 or self.config.n_movies <= 0:
            raise ValueError("users et movies doivent être positifs.")

    def _metadata(self):
        return {
            "n_users": self.config.n_users,
            "n_movies": self.config.n_movies,
            "n_ratings": self.config.n_ratings,
            "test_ratio": self.config.test_ratio,
            "random_state": self.config.random_state,
            "generator_version": "v5_stable_exact_ratings",
        }

    def generate_movies(self):
        genres = np.array([
            "Action",
            "Comedy",
            "Drama",
            "Sci-Fi",
            "Romance",
            "Thriller",
            "Animation",
            "Documentary",
        ])

        movie_numbers = np.arange(1, self.config.n_movies + 1)
        movie_ids = [f"M{i:05d}" for i in movie_numbers]

        genre_values = np.resize(genres, self.config.n_movies)
        self.rng.shuffle(genre_values)

        popularity_score = self.rng.random(self.config.n_movies)

        movies = pd.DataFrame({
            "movie_number": movie_numbers,
            "movie_id": movie_ids,
            "title": [f"Film {i:05d}" for i in movie_numbers],
            "genre": genre_values,
            "year": self.rng.integers(1990, 2025, size=self.config.n_movies),
            "popularity_score": popularity_score,
        })

        return movies

    def _ratings_per_user(self):
        """
        Répartit exactement n_ratings entre les utilisateurs.
        """
        counts = np.zeros(self.config.n_users, dtype=int)

        base_count = self.config.n_ratings // self.config.n_users
        remainder = self.config.n_ratings % self.config.n_users

        counts += base_count

        if remainder > 0:
            selected = self.rng.choice(
                self.config.n_users,
                size=remainder,
                replace=False,
            )
            counts[selected] += 1

        # Si possible, assurer au moins 2 notes par utilisateur présent.
        # Si n_ratings < 2*n_users, certains utilisateurs auront 0 note,
        # ce qui est acceptable pour un dataset sparse.
        if self.config.n_ratings >= 2 * self.config.n_users:
            counts[counts < 2] = 2
            extra = counts.sum() - self.config.n_ratings

            while extra > 0:
                candidates = np.where(counts > 2)[0]
                if len(candidates) == 0:
                    break

                chosen = self.rng.choice(candidates)
                counts[chosen] -= 1
                extra -= 1

        return counts

    def _weighted_sample_movies(self, candidate_numbers, size, popularity_scores):
        if size <= 0:
            return []

        candidate_numbers = np.array(candidate_numbers, dtype=int)

        if len(candidate_numbers) == 0:
            return []

        size = min(size, len(candidate_numbers))

        weights = popularity_scores[candidate_numbers - 1]
        weights = weights + 0.05
        weights = weights / weights.sum()

        return self.rng.choice(
            candidate_numbers,
            size=size,
            replace=False,
            p=weights,
        ).tolist()

    def _fill_missing_movies(self, selected, target_size, all_movie_numbers, popularity_scores):
        selected_set = set(selected)

        while len(selected) < target_size:
            remaining = np.array(
                [movie for movie in all_movie_numbers if movie not in selected_set],
                dtype=int,
            )

            if len(remaining) == 0:
                break

            needed = target_size - len(selected)
            extra = self._weighted_sample_movies(
                remaining,
                needed,
                popularity_scores,
            )

            for movie in extra:
                if movie not in selected_set:
                    selected.append(int(movie))
                    selected_set.add(int(movie))

        return selected

    def generate_ratings(self, movies):
        genres = movies["genre"].unique()
        all_movie_numbers = movies["movie_number"].to_numpy()
        popularity_scores = movies["popularity_score"].to_numpy()

        movie_id_by_number = dict(zip(movies["movie_number"], movies["movie_id"]))
        genre_by_number = dict(zip(movies["movie_number"], movies["genre"]))

        movies_by_genre = {
            genre: movies.loc[movies["genre"] == genre, "movie_number"].to_numpy()
            for genre in genres
        }

        counts = self._ratings_per_user()
        rows = []

        for user_index, n_user_ratings in enumerate(counts, start=1):
            if n_user_ratings <= 0:
                continue

            user_id = f"U{user_index:05d}"

            main_genre = self.rng.choice(genres)
            second_genre = self.rng.choice([g for g in genres if g != main_genre])

            n_main = int(n_user_ratings * 0.70)
            n_second = int(n_user_ratings * 0.20)
            n_random = n_user_ratings - n_main - n_second

            selected_movies = []

            selected_movies += self._weighted_sample_movies(
                movies_by_genre[main_genre],
                n_main,
                popularity_scores,
            )

            selected_set = set(selected_movies)

            second_pool = [
                movie for movie in movies_by_genre[second_genre]
                if movie not in selected_set
            ]

            selected_movies += self._weighted_sample_movies(
                second_pool,
                n_second,
                popularity_scores,
            )

            selected_set = set(selected_movies)

            random_pool = [
                movie for movie in all_movie_numbers
                if movie not in selected_set
            ]

            selected_movies += self._weighted_sample_movies(
                random_pool,
                n_random,
                popularity_scores,
            )

            selected_movies = self._fill_missing_movies(
                selected_movies,
                n_user_ratings,
                all_movie_numbers,
                popularity_scores,
            )

            self.rng.shuffle(selected_movies)

            for movie_number in selected_movies:
                movie_number = int(movie_number)
                movie_id = movie_id_by_number[movie_number]
                movie_genre = genre_by_number[movie_number]

                if movie_genre == main_genre:
                    rating = int(self.rng.choice([4, 5], p=[0.45, 0.55]))
                elif movie_genre == second_genre:
                    rating = int(self.rng.choice([3, 4, 5], p=[0.20, 0.45, 0.35]))
                else:
                    rating = int(
                        self.rng.choice(
                            [1, 2, 3, 4, 5],
                            p=[0.10, 0.20, 0.35, 0.25, 0.10],
                        )
                    )

                rows.append({
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": rating,
                    "timestamp": int(self.rng.integers(1_600_000_000, 1_750_000_000)),
                })

        ratings = pd.DataFrame(rows)

        # Sécurité : si un cas limite dépasse, on tronque.
        if len(ratings) > self.config.n_ratings:
            ratings = ratings.sample(
                n=self.config.n_ratings,
                random_state=self.config.random_state,
            ).reset_index(drop=True)

        return ratings

    def split_train_test(self, ratings):
        train_parts = []
        test_parts = []

        for _, group in ratings.groupby("user_id", sort=False):
            if len(group) < 2:
                train_parts.append(group)
                continue

            n_test = max(1, int(len(group) * self.config.test_ratio))

            positives = group[group["rating"] >= 4]
            others = group[group["rating"] < 4]

            if len(positives) >= n_test:
                test = positives.sample(
                    n=n_test,
                    random_state=int(self.rng.integers(0, 1_000_000)),
                )
            else:
                needed = n_test - len(positives)
                extra = others.sample(
                    n=min(needed, len(others)),
                    random_state=int(self.rng.integers(0, 1_000_000)),
                )
                test = pd.concat([positives, extra])

            train = group.drop(test.index)

            train_parts.append(train)
            test_parts.append(test)

        train = pd.concat(train_parts).reset_index(drop=True)

        if test_parts:
            test = pd.concat(test_parts).reset_index(drop=True)
        else:
            test = pd.DataFrame(columns=ratings.columns)

        return train, test

    def create_dataset(self):
        self._validate()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        movies = self.generate_movies()
        ratings = self.generate_ratings(movies)
        train, test = self.split_train_test(ratings)

        movies_to_save = movies.drop(columns=["movie_number"])

        movies_to_save.to_csv(self.data_dir / "movies.csv", index=False)
        ratings.to_csv(self.data_dir / "ratings.csv", index=False)
        train.to_csv(self.data_dir / "train.csv", index=False)
        test.to_csv(self.data_dir / "test.csv", index=False)

        with open(self.data_dir / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(self._metadata(), file, indent=2)

        return movies_to_save, ratings, train, test


class CsvDatasetRepository:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)

    def _expected_metadata(self):
        return {
            "n_users": self.config.n_users,
            "n_movies": self.config.n_movies,
            "n_ratings": self.config.n_ratings,
            "test_ratio": self.config.test_ratio,
            "random_state": self.config.random_state,
            "generator_version": "v5_stable_exact_ratings",
        }

    def _metadata_matches(self):
        metadata_path = self.data_dir / "metadata.json"

        if not metadata_path.exists():
            return False

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        return metadata == self._expected_metadata()

    def exists(self):
        required_files = [
            "movies.csv",
            "ratings.csv",
            "train.csv",
            "test.csv",
            "metadata.json",
        ]

        files_exist = all(
            (self.data_dir / file_name).exists()
            for file_name in required_files
        )

        return files_exist and self._metadata_matches()

    def load(self):
        movies = pd.read_csv(self.data_dir / "movies.csv")
        ratings = pd.read_csv(self.data_dir / "ratings.csv")
        train = pd.read_csv(self.data_dir / "train.csv")
        test = pd.read_csv(self.data_dir / "test.csv")

        return movies, ratings, train, test
