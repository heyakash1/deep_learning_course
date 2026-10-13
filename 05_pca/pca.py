import numpy as np


class PCA:
    """
    Principal Component Analysis, written with numpy only.

    Attributes learned by fit() end with an underscore (a scikit-learn habit):
        mean_                       shape (n_features,)
        components_                 shape (n_components, n_features), one component per row
        explained_variance_         shape (n_components,), variance along each component
        explained_variance_ratio_   shape (n_components,), fraction of the total variance
    """

    def __init__(self, n_components: int | None = None):
        """n_components: how many components to keep. None keeps all of them."""
        # Reject a nonsensical number right away, before any data is involved.
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")

        self.n_components = n_components

        # Everything below is filled in by fit(). None means "not fitted yet".
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        """
        X: array of shape (n_samples, n_features).
        learns: mean_, components_ (n_components, n_features),
                explained_variance_, explained_variance_ratio_
        returns: self
        """
        # Use float64 even if X is float32 (the fruit loader returns float32):
        # the covariance and eigenvalues need the extra precision.
        X = np.asarray(X, dtype=np.float64)

        # PCA needs a table: rows are examples, columns are features.
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")

        # Unpack the two sizes so they can be used by name below.
        n_samples, n_features = X.shape

        # The covariance divides by (n_samples - 1), so one sample is not enough.
        if n_samples < 2:
            raise ValueError("PCA needs at least 2 samples")

        # You cannot keep more components than there are features.
        if self.n_components is not None and self.n_components > n_features:
            raise ValueError(f"n_components={self.n_components} is larger than n_features={n_features}")

        # 1. Center the data.
        # Mean of every column: one number per feature, shape (n_features,).
        self.mean_ = X.mean(axis=0)
        # Subtract it from every row (numpy broadcasts the mean across all rows),
        # so every feature now averages 0. Shape stays (n_samples, n_features).
        X_centered = X - self.mean_

        # 2. Covariance matrix, shape (n_features, n_features).
        # Entry [i, j] says how features i and j vary together.
        # (n_features, n_samples) @ (n_samples, n_features) -> (n_features, n_features).
        cov_matrix = X_centered.T @ X_centered / (n_samples - 1)

        # 3. Eigen-decomposition of the symmetric covariance matrix.
        # eigenvalues: shape (n_features,), the variance along each direction.
        # eigenvectors: shape (n_features, n_features), the directions, one per COLUMN.
        # eigh is made for symmetric matrices: real values, perpendicular directions.
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort from the largest variance to the smallest.
        # eigh returns eigenvalues in ascending order, so argsort gives the positions
        # that sort them, and [::-1] reverses that to descending.
        sorted_indices = np.argsort(eigenvalues)[::-1]
        # Reorder the eigenvalues.
        eigenvalues = eigenvalues[sorted_indices]
        # Reorder the COLUMNS of the eigenvectors in the same way, so each direction
        # stays paired with its eigenvalue.
        eigenvectors = eigenvectors[:, sorted_indices]

        # A variance cannot be negative. Rounding errors can leave values like -2e-15
        # (for example when some pixels never vary), so clip those to exactly 0.
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        # Make the sign of each component reproducible. A direction and its opposite
        # describe the same axis, and eigh may return either one. Flip each column so
        # that its largest-magnitude entry is positive.
        # Row index of the largest |entry| in every column:
        largest_rows = np.argmax(np.abs(eigenvectors), axis=0)
        # The sign of that entry in every column (+1 or -1):
        signs = np.sign(eigenvectors[largest_rows, np.arange(n_features)])
        # An entry of exactly 0 has sign 0; treat it as +1 so no column is zeroed out.
        signs[signs == 0] = 1.0
        # Multiply every column by its sign (broadcast across the rows).
        eigenvectors = eigenvectors * signs

        # The total variance is the sum over ALL eigenvalues. Compute it before
        # dropping any component, so the ratios stay relative to the whole dataset.
        total_variance = eigenvalues.sum()
        # Fraction of the total carried by each component. If the total is 0
        # (every sample identical), avoid dividing by zero and report 0 for all.
        if total_variance > 0:
            explained_variance_ratio = eigenvalues / total_variance
        else:
            explained_variance_ratio = np.zeros_like(eigenvalues)

        # 5. Keep only the first n_components (None means keep everything).
        k = self.n_components if self.n_components is not None else n_features
        # Keep the first k columns of the eigenvectors, and the first k values of the rest.
        eigenvectors = eigenvectors[:, :k]
        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = explained_variance_ratio[:k]

        # Store the components as ROWS, shape (k, n_features). The eigenvectors are
        # columns, so transpose. This is the same layout scikit-learn uses.
        self.components_ = eigenvectors.T

        # Returning self lets calls chain, as in PCA(2).fit(X).transform(X).
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the components. returns shape (n_samples, n_components)."""
        # Fail with a clear message if fit() has not run yet.
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        # The data must have the same number of features as the data used in fit().
        if X.ndim != 2 or X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"X must have shape (n_samples, {self.mean_.shape[0]}), got {X.shape}")

        # Center with the mean learned in fit(), NOT the mean of this X: new data must
        # be centered on the same point as the training data.
        X_centered = X - self.mean_
        # Each score is the dot product of an example with a component.
        # (n_samples, n_features) @ (n_features, k) -> (n_samples, k).
        return X_centered @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map scores back to the original space. returns shape (n_samples, n_features)."""
        self._check_is_fitted()
        Z = np.asarray(Z, dtype=np.float64)

        # The scores must have one column per kept component.
        if Z.ndim != 2 or Z.shape[1] != self.components_.shape[0]:
            raise ValueError(f"Z must have shape (n_samples, {self.components_.shape[0]}), got {Z.shape}")

        # Rebuild each example as a weighted sum of the components,
        # (n_samples, k) @ (k, n_features) -> (n_samples, n_features),
        # then undo the centering by adding the mean back.
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X, then return its scores. returns shape (n_samples, n_components)."""
        return self.fit(X).transform(X)

    def _check_is_fitted(self) -> None:
        """Raise a clear error if fit() has not been called."""
        if self.components_ is None:
            raise RuntimeError("This PCA instance is not fitted yet. Call fit() first.")