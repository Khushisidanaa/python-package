# import numpy as np

# class ConformalPredictor:
#     """
#     A class used to represent a (Split) Conformal Predictor.

#     C_alpha(x) = [y^(x) +/- q_1-alpha(x)]

#     Parameters
#     ----------
#     regressor : object
#         A regressor object that has a 'predict' method.
#     alpha : float, default=0.05
#         The significance level used in the prediction interval calculation.

#     Attributes
#     ----------
#     scores : ndarray
#         The conformity scores of the calibration data.
#     quantile : float
#         The (1-alpha) empirical quantile of the conformity scores.

#     Methods
#     -------
#     fit(X, y)
#         Calibrates the conformal predictor using the provided calibration set.
#     predict(X)
#         Predicts the output for the given input `X` and provides a prediction interval.
#     """
#     def __init__(self, regressor, alpha=0.05):
#         self.regressor = regressor
#         self.alpha = alpha
#         self.scores = None
#         self.quantile = None

#     def fit(self, X, y):
#         """
#         Calibrates the conformal predictor using the provided calibration set.

#         Specifically, the `fit` method learns

#         q_1-alpha(x)

#         where q_1-alpha(x) is the (1-alpha) empirical quantile of the conformity scores

#         s = {|y_i - y^(x_i)|} U {infty}

#         Parameters
#         ----------
#         X : ndarray
#             The input data for calibration.
#         y : ndarray
#             The output data for calibration.
#         """
#         self.regressor.fit(X, y)
#         y_pred = self.regressor.predict(X)
#         self.scores = np.abs(y - y_pred)
#         self.quantile = np.quantile(self.scores, 1 - self.alpha, method='higher')

#     def predict(self, X):
#         """
#         Predicts the output for the given input `X` and provides a prediction interval.

#         C_alpha(x) = [y^(x) +/- q_1-alpha(x)]

#         Parameters
#         ----------
#         X : ndarray
#             The input data for which to predict the output.

#         Returns
#         -------
#         tuple
#             A tuple containing the prediction (1D `ndarray`) and the lower (1D `ndarray`)
#             and upper bounds (1D `ndarray`) of the prediction interval.
#         """
#         y_pred = self.regressor.predict(X)
#         lower_bound = y_pred - self.quantile
#         upper_bound = y_pred + self.quantile
#         return y_pred, lower_bound, upper_bound

import numpy as np

class ConformalPredictor:
    """
    A class used to represent a (Split) Conformal Predictor.

    C_alpha(x) = [y^(x) +/- q_1-alpha(x)]

    Parameters
    ----------
    regressor : object
        A regressor object that has a 'predict' method.
    alpha : float, default=0.05
        The significance level used in the prediction interval calculation.

    Attributes
    ----------
    scores : ndarray
        The conformity scores of the calibration data.
    quantile : float
        The (1-alpha) empirical quantile of the conformity scores.
    """

    def __init__(self, regressor, alpha=0.05):
        self.regressor = regressor
        self.alpha = alpha
        self.scores = None
        self.quantile = None

    def fit(self, X, y):
        """
        Calibrates the conformal predictor using the provided calibration set.

        Specifically, the `fit` method learns

        q_1-alpha(x)

        where q_1-alpha(x) is the (1-alpha) empirical quantile of the conformity scores

        s = {|y_i - y^(x_i)|} U {infty}

        Parameters
        ----------
        X : ndarray
            The input data for calibration.
        y : ndarray
            The output data for calibration.
        """
        self.regressor.fit(X, y)
        y_pred = self.regressor.predict(X)
        self.scores = np.abs(y - y_pred)
        self.quantile = np.quantile(self.scores, 1 - self.alpha, method='higher')

    def predict(self, X):
        """
        Predicts the output for the given input `X` and provides a prediction interval.

        C_alpha(x) = [y^(x) +/- q_1-alpha(x)]

        Parameters
        ----------
        X : ndarray
            The input data for which to predict the output.

        Returns
        -------
        tuple
            A tuple containing the prediction (1D `ndarray`) and the lower (1D `ndarray`)
            and upper bounds (1D `ndarray`) of the prediction interval.
        """
        y_pred = self.regressor.predict(X)
        lower_bound = y_pred - self.quantile
        upper_bound = y_pred + self.quantile
        return y_pred, lower_bound, upper_bound