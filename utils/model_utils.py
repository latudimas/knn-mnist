import numpy as np


class ModelUtils:
    @staticmethod
    def train_test_split(X, y, train_size, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        num_samples = len(X)
        indices = np.random.permutation(num_samples)
        train_samples = int(num_samples * train_size)

        train_indices = indices[:train_samples]
        test_indices = indices[train_samples:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def classification_report(y_true, y_pred):
        labels = np.unique(y_true)
        report = ""

        for label in labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            false_positive = np.sum((y_true != label) & (y_pred == label))
            false_negative = np.sum((y_true == label) & (y_pred != label))

            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0
                else 0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            report += f"Class {label}:\n"
            report += f"  Precision: {precision:.4f}\n"
            report += f"  Recall: {recall:.4f}\n"
            report += f"  F1-score: {f1:.4f}\n\n"

        return report

    @staticmethod
    def overall_classification_metrics(y_true, y_pred):
        labels = np.unique(y_true)
        precisions = []
        recalls = []
        f1_scores = []

        for label in labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            false_positive = np.sum((y_true != label) & (y_pred == label))
            false_negative = np.sum((y_true == label) & (y_pred != label))

            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0
                else 0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        overall_precision = np.mean(precisions)
        overall_recall = np.mean(recalls)
        overall_f1 = np.mean(f1_scores)

        return overall_precision, overall_recall, overall_f1

    @staticmethod
    def cross_val_score(model, X, y, cv=5):
        n_samples = len(X)
        fold_size = n_samples // cv
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        scores = []
        for i in range(cv):
            test_indices = indices[i * fold_size : (i + 1) * fold_size]
            train_indices = np.concatenate(
                [indices[: i * fold_size], indices[(i + 1) * fold_size :]]
            )

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = ModelUtils.accuracy_score(y_test, y_pred)
            scores.append(score)

        return np.array(scores)

    # @staticmethod
    # def knn_elbow_method(X, y, k_values=[3, 5, 7, 9, 11, 13, 15, 17, 19], cv=5):
    #     """
    #     Perform the Elbow method for KNN classification with specified K values.
    #
    #     Parameters:
    #     X (array-like): Feature matrix
    #     y (array-like): Target vector
    #     k_values (list): List of K values to evaluate
    #     cv (int): Number of folds for cross-validation
    #
    #     Returns:
    #     tuple: (k_values, accuracies)
    #     """
    #     from sklearn.neighbors import (
    #         KNeighborsClassifier,
    #     )  # Import here to keep sklearn dependency isolated
    #
    #     accuracies = []
    #
    #     for k in k_values:
    #         knn = KNeighborsClassifier(n_neighbors=k)
    #         scores = ModelUtils.cross_val_score(knn, X, y, cv=cv)
    #         accuracies.append(scores.mean())
    #
    #     # Plot the results
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(k_values, accuracies, "bo-")
    #     plt.xlabel("K value")
    #     plt.ylabel("Cross-validated accuracy")
    #     plt.title("Elbow Method for Optimal K in KNN")
    #     plt.xticks(k_values)  # Ensure all K values are shown on x-axis
    #     plt.grid(True)
    #     plt.show()
    #
    #     return k_values, accuracies
