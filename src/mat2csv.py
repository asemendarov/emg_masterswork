if __name__ == '__main__':

    import scipy.io as sio
    import pandas as pd

    file_name = 'data10mov_no_abs'

    mat = sio.loadmat('%s.mat' % file_name)

    data = pd.Series([value[0] for value in mat['data']])

    data.to_csv('%s.csv' % file_name)
