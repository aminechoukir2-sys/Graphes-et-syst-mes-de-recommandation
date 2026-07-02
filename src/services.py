from src.candidates import (
    BFSCandidateProvider,
    DFSCandidateProvider,
    DirectCandidateProvider,
)
from src.config import ProjectConfig
from src.dataset import CsvDatasetRepository, SyntheticMovieLensDatasetGenerator
from src.evaluators import CandidateEvaluator, RecommendationEvaluator
from src.graph import GraphBuilder
from src.recommenders import PageRankRecommender, RandomWalkRecommender


class DatasetService:
    def __init__(self, config: ProjectConfig, force_regenerate=False):
        self.config = config
        self.force_regenerate = force_regenerate
        self.repository = CsvDatasetRepository(config)
        self.generator = SyntheticMovieLensDatasetGenerator(config)

    def get_data(self):
        if not self.force_regenerate and self.repository.exists():
            return self.repository.load()

        return self.generator.create_dataset()


class RecommendationProjectService:
    def __init__(self, config: ProjectConfig, force_regenerate=False):
        self.config = config
        self.dataset_service = DatasetService(config, force_regenerate)
        self.graph_builder = GraphBuilder()

    def prepare(self):
        movies, ratings, train, test = self.dataset_service.get_data()
        graph = self.graph_builder.build(train)

        direct = DirectCandidateProvider(graph, train, self.config)
        bfs = BFSCandidateProvider(graph, train, self.config)
        dfs = DFSCandidateProvider(graph, train, self.config)

        providers = {
            "alone": direct,
            "bfs": bfs,
            "dfs": dfs,
        }

        recommenders = {
            "pagerank_alone": PageRankRecommender(graph, direct, self.config),
            "random_walk_alone": RandomWalkRecommender(graph, direct, self.config),
            "pagerank_bfs": PageRankRecommender(graph, bfs, self.config),
            "pagerank_dfs": PageRankRecommender(graph, dfs, self.config),
            "random_walk_bfs": RandomWalkRecommender(graph, bfs, self.config),
            "random_walk_dfs": RandomWalkRecommender(graph, dfs, self.config),
        }

        return {
            "movies": movies,
            "ratings": ratings,
            "train": train,
            "test": test,
            "graph": graph,
            "providers": providers,
            "recommenders": recommenders,
            "candidate_evaluator": CandidateEvaluator(test, self.config),
            "recommendation_evaluator": RecommendationEvaluator(test, self.config),
        }
