import pandas as pd
import numpy as np
import pickle
import json
import os

from pandas.io.json import json_normalize

def _load_data(data, n_prev = 10):
    """
    data should be pd.DataFrame()
    """
    # X = data.drop(['y', 'y_lab'], axis=1).as_matrix()
    X = data.drop(['y'], axis=1).as_matrix()
    # y = label_binarize(data['y'].values, [0, 1, 2])[:, :2]
    y = data['y'].values
    #y_lab = data['y_lab'].values
    
    docX, docY, docY_lab = [], [], []
    for i in range(len(data) - n_prev + 1):
        docX.append(X[i:i + n_prev])
        docY.append(y[i + n_prev - 1])
        #docY_lab.append(y_lab[i + n_prev -1])
    alsX = np.array(docX)
    alsY = np.array(docY)
    #alsY_lab = np.array(docY_lab)
    return alsX, alsY#, alsY_lab


def train_test_split(df, test_size=0.1, n_prev=10):  
    """
    This just splits data to training and testing parts
    """
   
    ntrn = int(len(df) * (1 - test_size))
    
    #X, y, y_lab = _load_data(df, n_prev=n_prev)

    X, y = _load_data(df, n_prev=n_prev)

    # X_train, y_train, y_train_lab = X[0:ntrn], y[0:ntrn], y_lab[0:ntrn]

    X_train, y_train = X[0:ntrn], y[0:ntrn]

    #X_test, y_test, y_test_lab = X[ntrn:], y[ntrn:], y_lab[ntrn:]

    X_test, y_test = X[ntrn:], y[ntrn:]

    return X_train, X_test, y_train, y_test#, y_train_lab, y_test_lab


# deprecated:
def cv_prepare_data(data):
    data['y'] = data.ref_pass
    data.drop(['ref_pass', 'frame_num', 'cam_num', 'cam_num.1', 'pass'], axis=1, inplace=True)

    return data


def parse_data(path_to_data='data/final_data/sensors_logs_correct_selected/'):
    """
    path_to_data : string
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

#depricated
def parse_data(dump=True):

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

    for pass_file in os.listdir('data/data_new/pass_sens/'):
            # print pass_file
            df = pd.DataFrame(columns=columns_names)

            with open('data/data_new/pass_sens/' + pass_file, 'r') as _file:
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

    data = [df for df in data_left]
    for df in data_right:
        data.append(df)

    if dump:
        with open('data/data_pkl.pkl', 'w') as fi:
            pickle.dump(data, fi)
        return 'dumped'
    else:
        return data


def load_data():
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

    data_new = []

    for i in np.random.permutation(len(data)):
        data_new.append(data[i])
    data = data_new

    conc_data = pd.concat(data)
    conc_data.set_index(['name', 'frame'], inplace=True)

    conc_data = conc_data.astype(int)

    return conc_data


#depricated
def load_new_data(shuffle=True):
    with open('data/data_pkl.pkl', 'r') as fi:
        data = pickle.load(fi)
    if shuffle:
        data_new = []
        for i in np.random.permutation(len(data)):
            data_new.append(data[i])
        data = data_new

    conc_data = pd.concat(data).drop(['sensors.left.ref_pass'], axis=1)
    conc_data['y'] = conc_data['sensors.right.ref_pass']
    conc_data.drop(['sensors.right.ref_pass'], inplace=True, axis=1)

    columns_ = [
        'l.cor',
        'l.loop',
        'l.pass',
        'l.shield',
        'r.cor',
        'r.loop',
        'r.pass',
        'r.shield',
        'y'
    ]

    conc_data.reset_index(inplace=True)
    conc_data.dropna(inplace=True)
    conc_data.set_index(['name', 'frame', 'index'], drop=True, inplace=True)

    conc_data = conc_data.astype(int)

    conc_data.columns = columns_

    return conc_data