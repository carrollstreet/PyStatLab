import numpy as np 

def mape(fact, predict):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    fact : array-like
        The actual values.
    predict : array-like
        The predicted values.

    Returns
    -------
    float
        The MAPE value.
    """
    fact, predict = np.asarray(fact), np.asarray(predict)
    return np.mean(np.abs((fact - predict) / fact))

def smape(fact, predict):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (sMAPE).

    Parameters
    ----------
    fact : array-like
        The actual values.
    predict : array-like
        The predicted values.

    Returns
    -------
    float
        The sMAPE value.
    """
    fact, predict = np.asarray(fact), np.asarray(predict)
    return np.mean(2*np.abs(fact - predict) / (np.abs(fact) + np.abs(predict)))
