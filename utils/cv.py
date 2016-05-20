import numpy as np
import scipy as sp

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from IPython.display import clear_output
from utils import init_weights, model_pq, morphological_filter

from theano_pq import pq_theano, pq_theano_f

from IPython.display import clear_output


def cv_build_model(X_train):

    model = Sequential()
    model.add(LSTM(input_dim=X_train.shape[2], output_dim=8, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))

    # or with regularized_mse loss
    model.compile(optimizer='rmsprop', loss='MSE')

    return model


def cv_fit_model(model, X_train, y_train, nb_epoch=20, weights=True):
    if weights:
        sample_weights = init_weights(y_train)
    else:
        sample_weights = np.ones_like(y_train)
    model.fit(X_train,
              y_train,
              batch_size=1024,
              nb_epoch=nb_epoch,
              shuffle=False,
              sample_weight=sample_weights)
    clear_output(wait=True)
    return model


def cv_threshold(X_train, y_train, model):
    result = sp.optimize.minimize_scalar(model_pq, bounds=[0.09, 0.95], args=(X_train, y_train, model), method='bounded')
    return result.x


def cv_predict(model, X, y, threshold):
    y_pred = (model.predict(X[:], batch_size=1024) > threshold).astype(int).flatten()
    y_pred = morphological_filter(y_pred, 5)
    return y_pred



# NEW _--------------------------------_

def cv_threshold(X_train, y_train, model, smooths = [0]):
    """
    Returs:
     smooth for best threshold
     threshold for this smooth
     """
    results = []
    for sm in smooths:
        y_pred = model.predict(X_train, batch_size=1024)
        results.append(sp.optimize.minimize_scalar(model_pq,
                                                   bounds=[0.09, 0.95],
                                                   args=(y_pred, y_train, sm),
                                                   method='bounded'))
    return results[np.argmin([result.fun for result in results])].x , smooths[np.argmin([result.fun for result in results])]

def cv_fit_model(model, X_train, y_train, nb_epoch=2, weights=False):
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

def cv_build_model(X_train, metrics=[pq_theano]):

    model = Sequential()
    model.add(LSTM(input_dim=X_train.shape[2], output_dim=8, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))

    # or with regularized_mse loss
    model.compile(optimizer='rmsprop',
                  loss='MSE',
                  metrics=metrics
                 )

    return model

