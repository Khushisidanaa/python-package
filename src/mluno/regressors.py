import numpy as np
from collections import Counter

class KNNRegressor:
    def __init__(self, k=5):
        """
        A class used to represent a K-Nearest Neighbors Regressor.

        Parameters
        ----------
        k : int, default=5
            The number of nearest neighbors to consider for regression.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the model using X as input data and y as target values.

        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape `(n_samples, 1)`
            where each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape `(n_samples, )`.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_new):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X_new : ndarray
            Input data, a 2D array of shape `(n_samples, 1)`, with which to make predictions.

        Returns
        -------
        ndarray
            The target values, which is a 1D array of shape `(n_samples, )`.
        """
        y_pred = np.zeros(X_new.shape[0])
        for i, x in enumerate(X_new):
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_neighbors = self.y_train[nearest_indices]
            y_pred[i] = np.mean(nearest_neighbors)

        return y_pred
    

class LinearRegressor:
    def __init__(self):
        """
        A class used to represent a Simple Linear Regressor.

        y = β0 + β1 * x + ε

        Attributes
        ----------
        weights : ndarray
            The weights of the linear regression model. Here, the weights are
            represented by the β vector which for univariate regression is a
            1D vector of length two, β = [β0, β1], where β0 is the slope and
            β1 is the intercept.
        """
        self.weights = None

    def fit(self, X, y):
        """
        Trains the linear regression model using the given training data.

        In other words, the `fit` method learns the weights, represented by
        the β vector. To learn the β vector, use:

        β^ = (X^T X)^-1 X^T y

        Here, X is the so-called design matrix, which, to include a term for
        the intercept, has a column of ones appended to the input `X` matrix.

        X = [1  x1
             1  x2
             ...
             1  xn]

        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape `(n_samples, 1)`
            where each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape `(n_samples, )`.
        """
        n_samples = X.shape[0]
        X_with_intercept = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
        self.weights = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)

    def predict(self, X):
        """
        Makes predictions for input data.

        y^ = X β^

        Parameters
        ----------
        X : ndarray
            Input data, a 2D array of shape `(n_samples, 1)`, with which to make predictions.

        Returns
        -------
        ndarray
            The predicted target values as a 1D array with the same length as `X`.
        """
        n_samples = X.shape[0]
        X_with_intercept = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
        y_pred = X_with_intercept @ self.weights
        return y_pred