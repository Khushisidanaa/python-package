import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(X, y, regressor, conformal=False, title=''):
    """
    Plot the predictions of a regressor on the given data.

    Parameters
    ----------
    X : ndarray
        The input data, a 2D array of shape `(n_samples, 1)`.
    y : ndarray
        The true target values, a 1D array of shape `(n_samples,)`.
    regressor : object
        A regressor object that has a 'predict' method.
    conformal : bool, default=False
        Whether to plot the prediction interval for a conformal predictor.
    title : str, default=''
        The title for the plot.

    Returns
    -------
    tuple
        A tuple containing the figure and axis objects of the plot.
    """
    if conformal:
        y_pred, lower_bound, upper_bound = regressor.predict(X)
    else:
        y_pred = regressor.predict(X)
    sorted_indices = np.argsort(X[:, 0])
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], y, color='blue', label='Data')
    ax.plot(X_sorted[:, 0], y_pred_sorted, color='red', label='Prediction')

    if conformal:
        lower_bound_sorted = lower_bound[sorted_indices]
        upper_bound_sorted = upper_bound[sorted_indices]
        ax.fill_between(X_sorted[:, 0], lower_bound_sorted, upper_bound_sorted, color='gray', alpha=0.2, label='Prediction Interval')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()

    return fig, ax
