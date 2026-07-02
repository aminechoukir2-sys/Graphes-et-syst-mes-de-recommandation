from abc import ABC, abstractmethod


class BaseDatasetGenerator(ABC):
    @abstractmethod
    def create_dataset(self):
        pass


class BaseGraph(ABC):
    @abstractmethod
    def add_edge(self, user_id, movie_id, weight):
        pass

    @abstractmethod
    def build_from_ratings(self, ratings):
        pass

    @abstractmethod
    def get_neighbors(self, node):
        pass

    @abstractmethod
    def nodes(self):
        pass


class BaseCandidateProvider(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def find_candidate_movies(self, user_id):
        pass

    @abstractmethod
    def local_nodes(self, user_id):
        pass


class BaseRecommender(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def score_items(self, user_id):
        pass

    def recommend(self, user_id, top_k):
        scores = self.score_items(user_id)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def calculate(self, recommended_items, relevant_items, k):
        pass
