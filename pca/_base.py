import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None

    def fit(self, X):
        # Standardize the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculate covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvectores by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components
        self.components = eigenvectors[:, : self.n_components]
        self.eigenvalues = eigenvalues[: self.n_components]

    def transform(self, X):
        # project data onto principal components
        if self.mean is None or self.components is None:
            raise ValueError(
                "PCA has not been fitted. Call fit() before using transform()."
            )

        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def inverse_transform(self, X_transformed):
        if self.components is not None and self.mean is not None:
            return np.dot(X_transformed, self.components.T) + self.mean
        else:
            raise ValueError(
                "Components and/or mean have not been calculated. Please fit the PCA object first."
            )

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
