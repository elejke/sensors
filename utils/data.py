import pandas as pd

def load_data():
    new_pd_data = pd.read_csv('data/new_pd_data_full_int.csv')
    new_pd_data = new_pd_data.drop(['frame_num.1', 'Unnamed: 0', 'ref_pass.1'], axis=1)
    return new_pd_data