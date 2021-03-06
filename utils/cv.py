import numpy as np
import scipy as sp

from IPython.display import clear_output
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from utils import init_weights, morphological_filter
from pass_quality import approx_pq


def cv_build_model(X_train):
    """
    A function for building Recurrent Neural Network model with
    predefined parameters (X_train used because of input_shape
    parameter estimation)

    :param X_train: training dataset
    :return: model: keras neural netwok model
    """
    model = Sequential()
    model.add(LSTM(input_dim=X_train.shape[2], output_dim=8, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))

    # or with regularized_mse loss
    model.compile(optimizer='rmsprop',
                  loss='MSE',
                  )
    return model


def cv_fit_model(model, X_train, y_train, nb_epoch=2, weights=False):
    """
    Fit RNN model with ...
    :param model:
    :param X_train:
    :param y_train:
    :param nb_epoch:
    :param weights:
    :return:
    """

    if weights:
        sample_weights = init_weights(y_train)
    else:
        sample_weights = np.ones_like(y_train)
    for i in range(nb_epoch):
        print "Epoch " + str(i) + '/' + str(nb_epoch)
        model.fit(X_train,
                  y_train,
                  batch_size=1024,
                  nb_epoch=1,
                  shuffle=False,
                  sample_weight=sample_weights
                 )
        clear_output(wait=True)
    return model


def model_pq(threshold, y_pred, y_train, sm):
    """
    Smoothed model pass_quality function estimator.

    :param threshold:
    :param y_pred:
    :param y_train:
    :param sm:
    :return:
    """
    y_pred = (y_pred > threshold).astype(int).flatten()
    return -approx_pq(y_train, morphological_filter(y_pred, sm))[0]


def cv_threshold(X_train, y_train, model, smooths=[0, 2, 3]):
    """
    Find threshold and best smooth size
    :param X_train:
    :param y_train:
    :param model:
    :param smooths:
    :return:
    """
    results = []
    for sm in smooths:
        y_pred = model.predict(X_train, batch_size=1024)
        results.append(sp.optimize.minimize_scalar(model_pq,
                                                   bounds=[0.09, 0.95],
                                                   args=(y_pred, y_train, sm),
                                                   method='bounded'))
    return (results[np.argmin([result.fun for result in results])].x,
            smooths[np.argmin([result.fun for result in results])])
