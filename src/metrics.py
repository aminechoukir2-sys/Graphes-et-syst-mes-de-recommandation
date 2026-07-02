from src.interfaces import BaseMetric


class PrecisionAtK(BaseMetric):
    @property
    def name(self):
        return "precision_at_k"

    def calculate(self, recommended_items, relevant_items, k):
        recommended_items = recommended_items[:k]
        return len(set(recommended_items) & set(relevant_items)) / k


class RecallAtK(BaseMetric):
    @property
    def name(self):
        return "recall_at_k"

    def calculate(self, recommended_items, relevant_items, k):
        if not relevant_items:
            return None

        recommended_items = recommended_items[:k]
        return len(set(recommended_items) & set(relevant_items)) / len(relevant_items)


class F1AtK(BaseMetric):
    @property
    def name(self):
        return "f1_at_k"

    def calculate(self, recommended_items, relevant_items, k):
        precision = PrecisionAtK().calculate(recommended_items, relevant_items, k)
        recall = RecallAtK().calculate(recommended_items, relevant_items, k)

        if recall is None:
            return None

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)
