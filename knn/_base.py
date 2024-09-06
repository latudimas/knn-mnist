from typing import Counter
import numpy as np


class KNN:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, x, y) -> None:
        self.X_train = x
        self.y_train = y

    def euclidian_distance(self, x1, x2) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X) -> np.ndarray:
        return np.array([self._predict(x) for x in X])

    def _predict(self, x: np.ndarray) -> str:
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must fit the model before making predictions.")

        # Compute distances
        distances = [self.euclidian_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of k nearest neighbors
        k_indices = np.argsort(distances)[: self.k]

        # Get the labels of k nearest neighbors
        k_nearest_labels = [self.y_train[1] for i in k_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
