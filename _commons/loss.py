import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


def MAE(pred_y, true_y):
    return np.mean(np.abs(pred_y - true_y))


def MSE(pred_y, true_y):
    return np.mean((pred_y - true_y) ** 2)


def MAPE(pred_y, true_y):
    return mean_absolute_percentage_error(true_y, pred_y) * 100
    # return np.mean(np.abs((pred_y - true_y) / true_y))
