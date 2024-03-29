import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.svm as svm
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

import ctypes.wintypes

CSIDL_PERSONAL = 5       # My Documents
SHGFP_TYPE_CURRENT = 0   # Get current, not default value

buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)

# global variable
documents_path = buf.value
project_relative_path = 'JetBrains PyCharm/emg_masterswork'
file_relative_path = 'data/data10mov_no_abs.mat'

test_size = 0.33
split_random_state = 1
search_random_state = 10
classes = '0, 1, 4, 5, 6, 7'
# classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

svc = svm.SVC(kernel='poly')

algorithms = {
    'Standard Scaler': StandardScaler(),
    'Min-Max Scaler': MinMaxScaler(),
    'Max-Abs Scaler': MaxAbsScaler(),
    'Normalizer': Normalizer(),
    'Robust Scaler': RobustScaler(quantile_range=(25, 75)),
    'Yeo-Johnson Power Transformer': PowerTransformer(method='yeo-johnson'),
    'Uniform Quantile Transformer': QuantileTransformer(output_distribution='uniform'),
    'Normal Quantile Transformer': QuantileTransformer(output_distribution='normal'),
}

parameters = {
    # 'linear': {
    #     'C': np.arange(0.5, 10.1, 0.5),
    # },
    'rbf': {
        'C': np.arange(1, 5.1, 1),
        'gamma': np.arange(0.01, 0.2, 0.01),
    },
    # 'sigmoid': {
    #     'C': np.arange(0.5, 5.1, 0.5),
    #     'gamma': np.arange(0.01, 0.5, 0.01),
    #     'coef0': np.arange(1, 5),
    # },
    'poly': {
        'C': np.arange(1, 5.1, 1),
        'gamma': np.arange(0.01, 0.1, 0.01),
        'coef0': np.arange(1, 5),
        'degree': np.arange(1, 5)
    },
}

clf = RandomizedSearchCV(svc, parameters['poly'], n_iter=10,
                         cv=5, return_train_score=True, iid=False,
                         random_state=search_random_state)


def read_mat(*paths):
    mat = sio.loadmat(os.path.join(documents_path, *paths))
    data = pd.Series([value[0] for value in mat['data']], [
        'Кисть вверх',                  # 0
        'Кисть вниз',                   # 1
        'Сжатие всех пальцев',          # 2
        'Сжатие указ пальца',           # 3
        'Сжатие среднего пальца',       # 4
        'Сжатие безымянного пальца',    # 5
        'Щелчок большого с средним',    # 6
        'Разжимание всех пальцев',      # 7
        'Поворот кисти влево',          # 8
        'Поворот кисти вправо'          # 9
    ])
    return data


def preprocessing_data(algorithm, X_train, X_test):
    return algorithm.fit_transform(X_train), algorithm.transform(X_test)


def train_and_test(X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return clf.score(X_train, y_train), clf.score(X_test, y_test)


def _X(data):
    return np.vstack(data)


def _y(classes, data_len):
    return np.ravel(np.array(classes * data_len).reshape((data_len, -1)).T)


def print_dict(data: dict, begin=None, sep=None, end=None):
    print(
        begin or "",
        pd.Series(data),
        sep=sep or "\n",
        end=end or '-------------------------------------\n'
    )


def main():
    emg_data_set = read_mat(project_relative_path, file_relative_path)

    classes_list = list(set(map(int, classes.split(","))))

    emg_data_set = emg_data_set.iloc[classes_list]

    X, y = _X(emg_data_set), _y(classes_list, len(emg_data_set[0]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state)

    new_X_train, new_X_test = [], []

    result = list()

    for kernel, params in parameters.items():
        svc.kernel = kernel

        for name, algorithm in algorithms.items():
            start_time = time.time()
            new_X_train, new_X_test = preprocessing_data(algorithm, X_train, X_test)
            train_results, tests_results = train_and_test(new_X_train, y_train, new_X_test, y_test)
            df_results = pd.DataFrame(clf.cv_results_)
            df_results = df_results.loc[df_results['mean_test_score'].idxmax()][['params']]
            result.append(
                {
                    'Kernel': kernel,
                    'Preprocessing method': name,
                    'Train result': f"{train_results:.3%}",
                    'Test result': f"{tests_results:.3%}",
                    'Params': dict(pd.Series(df_results[0]).loc[params.keys()]),
                    'Time': time.time() - start_time,
                }
            )
            print_dict(result[-1])

    result = sorted(result, key=lambda kv: kv['Test result'])

    print_dict(result[-1], begin='\n\n!!! Лучший результат:')


if __name__ == '__main__':
    main()
