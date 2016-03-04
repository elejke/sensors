
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.optimizers import RMSprop


# In[2]:

def split_by_passes(y):
    if len(y) > 1:
        diffs = np.diff(y)
    else:
        diffs = [0]

    passes_starts, passes_ends = [], []
    
    if y[0] == 1:
        passes_starts.append(0)
    
    for num, diff in enumerate(diffs):
        if diff == 1:
            passes_starts.append(num + 1)
        if diff == -1:
            passes_ends.append(num + 1)

    if y[-1] == 1:
        passes_ends.append(len(y))

    return [(passes_starts[i], passes_ends[i]) for i in range(len(passes_starts))]


# In[3]:

def PQ(y_true, y_pred):
    ###
    
    #R_est = R(y_true, y_pred)
    #I_est = I(y_true, y_pred)
    #Gh_est = Gh(y_true, y_pred)
    #return float(R_est)/(I_est + Gh_est)
    pass


# In[4]:

def pass_quality(y_true, y_pred):
    """Nikolaev, Malugina"""
    
    R_est = R(y_true, y_pred)
    I_est = I(y_true, y_pred)
    Gh_est = Gh(y_true, y_pred)
    return float(R_est)/(I_est + Gh_est)


# In[5]:

def R(y_true, y_pred):
    right = []
    for i, j in split_by_passes(y_true):
        right.append(len(split_by_passes(y_pred[i:j])) == 1)
    return np.count_nonzero(right)


# In[6]:

def I(y_true, y_pred):
    return len(split_by_passes(y_true))


# In[7]:

def Gh(y_true, y_pred):
    right = []
    for i, j in split_by_passes(y_pred):
        right.append(int(len(split_by_passes(y_true[i:j])) == 0))
    return np.count_nonzero(right)


def prepare_seq_data(X, seq_len=5):
    return np.stack([X.values[seq_len-i:-i] for i in range(1, seq_len+1)], axis=1)

def padding(X, y, seq_len=5):
    return np.stack([X.values[seq_len-i:-i] for i in range(5, 0, -1)], axis=1), y[seq_len-1:]
    #return np.stack([X.values[i:-(seq_len-i)] for i in range(seq_len)], axis=1)


