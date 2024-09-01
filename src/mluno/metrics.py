import numpy as np

def rmse(y_true, y_pred):
    """
    Compute the Root Mean Square Error (RMSE).

    Parameters
    ----------
    y_true : ndarray
        A 1D array of the true target values.
    y_pred : ndarray
        A 1D array of the predicted target values.

    Returns
    -------
    float
        The RMSE between the true and predicted target values.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_value = np.sqrt(mse)
    return rmse_value

def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : ndarray
        A 1D array of the true target values.
    y_pred : ndarray
        A 1D array of the predicted target values.

    Returns
    -------
    float
        The MAE between the true and predicted target values.
    """
    abs_errors = np.abs(y_true - y_pred)
    mae_value = np.mean(abs_errors)
    return mae_value

def coverage(y_true, y_lower, y_upper):
    """
    Calculate the empirical coverage of the prediction intervals.

    Parameters
    ----------
    y_true : ndarray
        The true target values.
    y_lower : ndarray
        The lower bounds of the prediction intervals.
    y_upper : ndarray
        The upper bounds of the prediction intervals.

    Returns
    -------
    float
        The empirical coverage, which is the proportion of true target values
        that fall within the predicted intervals.
    """
    within_interval = np.logical_and(y_lower <= y_true, y_true <= y_upper)
    empirical_coverage = np.mean(within_interval)
    return empirical_coverage

def sharpness(y_pred_lower, y_pred_upper):
    """
    Compute the sharpness of the prediction intervals.

    Parameters
    ----------
    y_pred_lower : ndarray
        A 1D array of lower bounds of the predicted intervals.
    y_pred_upper : ndarray
        A 1D array of upper bounds of the predicted intervals.

    Returns
    -------
    float
        The average width of the predicted intervals.
    """
    sharpness = np.mean(y_pred_upper - y_pred_lower)
    return sharpness
