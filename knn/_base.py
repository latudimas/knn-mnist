from typing import Counter
import numpy as np


class KNNClassifier:
    def __init__(self, n_neighbors=3, metric="euclidean", p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must fit the model before making predictions.")

        distances = np.array(
            [self._calculate_distance(x, x_train) for x_train in self.X_train]
        )
        k_indices = np.argsort(distances)[: self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _calculate_distance(self, x1, x2):
        match self.metric:
            case "euclidean":
                return self._euclidian_distance(x1, x2)
            case "manhattan":
                return self._manhattan_distance(x1, x2)
            case "minkowski":
                return self._minkowski_distance(x1, x2)
            case _:
                raise ValueError(f"Unsupported distance metric '{self.metric}'")

    def _euclidian_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        diff = np.abs(x1 - x2)
        return np.sum(diff)

    def _minkowski_distance(self, x1, x2):
        abs_diff = np.abs(x1 - x2)
        distance = np.sum(abs_diff**self.p) ** (1.0 / self.p)
        return distance

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    # def predict_proba(self, X):
    #     if self.X_train is None or self.y_train is None:
    #         raise ValueError(
    #             "Model not fitted. Call 'fit' before using 'predict_proba'."
    #         )
    #
    #     X = np.asarray(X)
    #     classes = np.unique(self.y_train)
    #     n_samples, n_features = X.shape
    #
    #     # Pre-allocate the probabilities array
    #     probas = np.zeros((n_samples, len(classes)))
    #
    #     for i, x in enumerate(X):
    #         distances = self._calculate_distances(self.X_train, x)
    #         k_indices = np.argpartition(distances, self.n_neighbors)[: self.n_neighbors]
    #         k_nearest_labels = self.y_train[k_indices]
    #         class_counts = Counter(k_nearest_labels)
    #
    #         for j, c in enumerate(classes):
    #             probas[i, j] = class_counts[c] / self.n_neighbors
    #
    #     return probas

    # def predict_proba(self, X):
    #     if not isinstance(X, np.ndarray):
    #         X = np.array(X)
    #
    #     probas = []
    #     classes = np.unique(self.y_train)
    #     for x in X:
    #         distances = self._calculate_distances(x)
    #         k_indices = np.argpartition(distances, self.n_neighbors)[: self.n_neighbors]
    #         k_nearest_labels = self.y_train[k_indices]
    #         class_counts = Counter(k_nearest_labels)
    #         proba = [class_counts[c] / self.n_neighbors for c in classes]
    #         probas.append(proba)
    #     return np.array(probas)
