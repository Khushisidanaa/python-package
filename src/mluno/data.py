import numpy as np
def make_line_data(n_samples=100, beta_0=0, beta_1=1, sd=1, X_low=-10, X_high=10, random_seed=None):
   """
    Generate data for linear regression.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    beta_0 : float, default=0
        The true intercept of the linear model.
    beta_1 : float, default=1
        The true slope of the linear model.
    sd : float, default=1
        Standard deviation of the normally distributed errors.
    X_low : float, default=-10
        Lower bound for the uniform distribution of X.
    X_high : float, default=10
        Upper bound for the uniform distribution of X.
    random_seed : int, optional
        Seed to control randomness.

    Returns
    -------
    tuple
        A tuple containing the `X` and `y` arrays. `X` is a 2D array with shape `(n_samples, 1)`
        and `y` is a 1D array with shape `(n_samples,)`. `X` contains the simulated `X` values
        and `y` contains the corresponding true mean of the linear model with added normally
        distributed errors.
    """
   
   if random_seed is not None:
      np.random.seed(random_seed)
   X = np.random.uniform(X_low, X_high, size=(n_samples, 1))
   y = beta_0 + beta_1 * X.ravel() + np.random.normal(loc=0, scale=sd, size=n_samples)
   return X, y

def make_sine_data(n_samples=100, sd=1, X_low=-6, X_high=6, random_seed=None):
   """
    Generate data for nonlinear regression.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    sd : float, default=1
        Standard deviation of the normally distributed errors.
    X_low : float, default=-6
        Lower bound for simulated `X` values.
    X_high : float, default=6
        Upper bound for simulated `X` values.
    random_seed : int, optional
        Seed to control randomness.

    Returns
    -------
    tuple
        A tuple containing the `X` and `y` arrays. `X` is a 2D array with shape `(n_samples, 1)`
        and `y` is a 1D array with shape `(n_samples,)`. `X` contains the simulated `X` values
        and `y` contains the corresponding sine values with added normally distributed errors.
    """
   if random_seed is not None:
      np.random.seed(random_seed)
   X = np.random.uniform(X_low, X_high, size=(n_samples, 1))
   y = np.sin(X.ravel()) + np.random.normal(0, sd, size=n_samples)
   return X, y

def split_data(X, y, holdout_size=0.2, random_seed=None):
    """
    Split the data into train and test sets.

    Parameters
    ----------
    X : ndarray
        The feature data to be split. A 2D array with shape `(n_samples, 1)`.
    y : ndarray
        The target data to be split. A 1D array with shape `(n_samples,)`.
    holdout_size : float, default=0.2
        The proportion of the data to be used as the test set.
    random_seed : int, optional
        Seed to control randomness.

    Returns
    -------
    tuple
        The split train and test data: `(X_train, X_test, y_train, y_test)`.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    test_size = int(np.floor(holdout_size * n_samples))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test