import pandas as pd
import numpy as np
import pickle
import json
import os

from pandas.io.json import json_normalize


def add_prev(data, n_prev=45):
    """
    Addition of n_prev previous X steps.
    :param data: pd.DataFrame
    :param n_prev: int,
    :return:
    """
    X = data.drop(['y'], axis=1).as_matrix()
#    X_pad = np.zeros(X.shape[-1] * (n_prev - 1)).reshape(n_prev - 1, X.shape[-1])
#    X = np.concatenate([X_pad, X], axis=0)

#    y_pad = np.zeros(n_prev - 1)
    y = data['y'].values
#    y = np.concatenate([y_pad, y], axis=0)

    doc_x, doc_y = [], []

    for i in range(len(X) - n_prev + 1):
        doc_x.append(X[i:i + n_prev])
        doc_y.append(y[i + n_prev - 1])

    als_x = np.array(doc_x)
    als_y = np.array(doc_y)
    return als_x, als_y


def train_test_split(data, test_size=0.25, n_prev=45):
    """
    Train and test data splitting function
    :param data: pd.DataFrame
    :param test_size: float, len(X_test) / len(data)
    :param n_prev: int,
    :return: splitted data
    """
    n_train = int(len(data) * (1 - test_size))
    X, y = add_prev(data, n_prev=n_prev)
    X_train, y_train = X[0:n_train], y[0:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    return X_train, X_test, y_train, y_test


def parse_data(path_to_data='data/final_data/sensors_logs_correct_selected/'):
    """
    Parse data from directory with logs
    :param path_to_data: path to directory
    """
    columns_names = [
        'frame',
        'sensors.left.cor',
        'sensors.left.loop',
        'sensors.left.pass',
        'sensors.left.ref_pass',
        'sensors.left.shield',
        'sensors.right.cor',
        'sensors.right.loop',
        'sensors.right.pass',
        'sensors.right.ref_pass',
        'sensors.right.shield',
        'name'
    ]

    data_left = []
    data_right = []

    for pass_file in os.listdir(path_to_data):
            # print pass_file
            df = pd.DataFrame(columns=columns_names)

            with open(path_to_data + pass_file, 'r') as _file:
                for line in _file:
                    df = df.append(json_normalize(json.loads(line)))
                    df.name = pass_file
                    if 't' in df.columns:
                        df.drop(['t'], axis=1, inplace=True)
                    if 'sensors.left' in df.columns:
                        df.drop(['sensors.left'], axis=1, inplace=True)
                    if 'sensors.right' in df.columns:
                        df.drop(['sensors.right'], axis=1, inplace=True)

                    #df['y'] = df['sensors.left.ref_pass'] | df['sensors.right.ref_pass']

                #df.drop(['sensors.left.ref_pass', 'sensors.right.ref_pass'], axis=1, inplace=True)
                if 'left' in pass_file:
                    data_left.append(df.reset_index(drop=True))
                else:
                    data_right.append(df.reset_index(drop=True))
            _file.close
    with open('data/final_data_left.pkl', 'wb') as fi:
        pickle.dump(data_left, fi)
    with open('data/final_data_right.pkl', 'wb') as fi:
        pickle.dump(data_right, fi)


def load_data():
    """
    Loading of pickled data, permute and concatenate it
    :return:
    """
    with open('data/final_data_left.pkl', 'rb') as fi:
        data_left = pickle.load(fi)
    with open('data/final_data_right.pkl', 'rb') as fi:
        data_right = pickle.load(fi)

    data_left_new = []

    for df in data_left:
        df1 = df[['name',
                  'frame',
                  'sensors.left.shield',
                  'sensors.left.loop',
                  'sensors.left.cor',
                  'sensors.left.pass',
                  'sensors.left.ref_pass',
                 ]]
        df1.columns = ['name', 'frame', 'shield', 'loop', 'cor', 'pass', 'y']
        data_left_new.append(df1)

        data_right_new = []

    for df in data_right:
        df1 = df[['name',
                  'frame',
                  'sensors.right.shield',
                  'sensors.right.loop',
                  'sensors.right.cor',
                  'sensors.right.pass',
                  'sensors.right.ref_pass',
                 ]]
        df1.columns = ['name', 'frame', 'shield', 'loop', 'cor', 'pass', 'y']
        data_right_new.append(df1)

    data = [df for df in data_left_new]
    for df in data_right_new:
        data.append(df)

    #data = [df for df in data_left_new]

    data_new = []

    for i in np.random.permutation(len(data)):
            data_new.append(data[i])
            data_new.append(pd.DataFrame(data=[np.zeros(len(data[i].columns))], columns=data[i].columns))
    data = data_new

    conc_data = pd.concat(data)
    conc_data.set_index(['name', 'frame'], inplace=True)

    conc_data = conc_data.astype(int)

    return conc_data
