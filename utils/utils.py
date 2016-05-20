import numpy as np
import matplotlib.pylab as plt

import theano.tensor as tt

from keras.objectives import MSE
from pass_quality import approx_pq


def plot_pq(y_true, y_pred, step=1024):
    pqs = []
    for i in range(len(y_true) / step):
        pqs.append(approx_pq(y_true[:i * step + step], y_pred[:i * step + step])[0])

    plt.plot(np.arange(len(y_true) / step), pqs)


# post_processing
def morphological_filter(y_pred, n_times):
    """
    Apply morphological filter to any series of 0 and 1
    n_times: integer:
        number of applies
    """
    for i in range(n_times):
        y_pred = np.concatenate([
            [y_pred[0] + y_pred[1] >= 1],
            [np.mean([y_pred[i - 1],
                      y_pred[i],
                      y_pred[i + 1]]) > 0.5 for i in xrange(1, len(y_pred) - 1)],
            [y_pred[-1] + y_pred[-2] >= 1]]).astype(int)
    return y_pred


def model_pq(threshold, y_pred, y_train, sm):
    y_pred = (y_pred > threshold).astype(int).flatten()
    return -approx_pq(y_train, morphological_filter(y_pred, sm))[0]


def regularized_mse(y_true, y_pred):
    # compute Mean Squared Error:
    mean_squared_error = MSE(y_true, y_pred)

    mean_squared_gradient = tt.mean((tt.extra_ops.diff(y_pred.T[-1].T)) ** 2)

    return mean_squared_error * (7. / 8) + mean_squared_gradient * (1. / 8)


def init_weights(y_train):
    weights = np.zeros_like(y_train) + 0.4
    for ind in range(len(y_train[2:-2])):
        if y_train[ind] == 1 and y_train[ind - 1] == 0 and y_train[ind - 2] == 0:
            weights[ind + 2] += 0.2
            weights[ind + 1] += 0.4
            weights[ind] += 0.6
            weights[ind - 1] += 0.4
            weights[ind - 2] += 0.2
        if y_train[ind] == 0 and y_train[ind - 1] == 1 and y_train[ind - 2] == 1:
            weights[ind + 2] += 0.2
            weights[ind + 1] += 0.4
            weights[ind] += 0.6
            weights[ind - 1] += 0.4
            weights[ind - 2] += 0.2
        if y_train[ind] == 0 and y_train[ind + 1] == 1 and y_train[ind - 1] == 1:
            weights[ind + 2] += 0.2
            weights[ind + 1] += 0.4
            weights[ind] += 0.6
            weights[ind - 1] += 0.4
            weights[ind - 2] += 0.2
    weights = weights / np.mean(weights)
    return weights