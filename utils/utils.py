import numpy as np
from pass_quality import pass_quality, approx_pq


# post_processing
def morphological_filter(y_pred, n_times):
    """
    Apply morphological filter to any series of 0 and 1

    :param y_pred:
    :param n_times:
    :return:
    """
    for i in range(n_times):
        y_pred = np.concatenate([
            [y_pred[0] + y_pred[1] >= 1],
            [np.mean([y_pred[i - 1],
                      y_pred[i],
                      y_pred[i + 1]]) >= 0.5 for i in xrange(1, len(y_pred) - 1)],
            [y_pred[-2] + y_pred[-1] >= 1]]).astype(int)
    return y_pred


# post_processing
def morphological_filter_fair(y_pred, n_times):
    """
    Apply morphological filter to any series of 0 and 1

    :param y_pred:
    :param n_times:
    :return:
    """
    for i in range(n_times):
        y_pred = np.concatenate([
            [y_pred[0] + y_pred[1] >= 1],
            [y_pred[1] + y_pred[2] >= 1],
            [np.mean([y_pred[i - 2],
                      y_pred[i - 1],
                      y_pred[i]]) >= 0.5 for i in xrange(2, len(y_pred))]]).astype(int)
            #[y_pred[-1] + y_pred[-2] >= 1]]).astype(int)
    return y_pred


def init_weights(y_train):
    """
    Initialize weights for y_train for RNN training.

    :param y_train:
    :return:
    """
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
