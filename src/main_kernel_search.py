import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.svm as svm
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# global variable
# C = 4
# gamma = 0.02
test_size = 0.33
split_random_state = 1
# model = svm.SVC(kernel='rbf', gamma=gamma, C=C)

# Выбор классов
# classes = '0, 1, 4, 5, 7'
# classes = '0, 1, 4, 6, 7'
classes = '0, 1, 4, 7, 9'
# classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

file_name_data_set = '../data/data10mov_no_abs.mat'

qt = QuantileTransformer()

search_random_state = 10
svc = svm.SVC()

algorithm = QuantileTransformer(output_distribution='uniform')

parameters = {
    'linear': {
        'C': np.arange(0.5, 10.1, 0.5),
    },
    'rbf': {
        'C': np.arange(0.5, 5.1, 0.5),
        'gamma': np.arange(0.01, 0.5, 0.01),
    },
    'sigmoid': {
        'C': np.arange(0.5, 5.1, 0.5),
        'gamma': np.arange(0.01, 0.5, 0.01),
        'coef0': np.arange(1, 5),
    },
    'poly': {
        'C': np.arange(0.5, 5.1, 0.5),
        'gamma': np.arange(0.01, 0.5, 0.01),
        'coef0': np.arange(1, 5),
        'degree': np.arange(1, 5)
    },
}

clf = RandomizedSearchCV(svc, parameters['rbf'], cv=5, iid=False,
                         random_state=search_random_state)

# test global variable
kernel_list = ('linear', 'poly', 'rbf', 'sigmoid')


def read_mat(file_name):
    mat = sio.loadmat(file_name)
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


def convert_to_gram_matrix(X_train, X_test):
    X_train_T = np.array(X_train).T
    return np.dot(X_train, X_train_T), np.dot(X_test, X_train_T)


def train_and_test(X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return clf.score(X_train, y_train), clf.score(X_test, y_test)


def _X(data):
    return np.vstack(data)


def _y(classes, data_len):
    return np.ravel(np.array(classes * data_len).reshape((data_len, -1)).T)


def main(print_result=True) -> list:
    emg_data_set = read_mat(os.path.abspath(file_name_data_set))

    classes_list = list(set(map(int, classes.split(","))))

    emg_data_set = emg_data_set.iloc[classes_list]

    X, y = _X(emg_data_set), _y(classes_list, len(emg_data_set[0]))

    start_time = time.time()                            # time begin (optional)

    result = dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state)

    X_train, X_test = preprocessing_data(algorithm, X_train, X_test)

    for kernel in kernel_list:
        svc.kernel = kernel
        clf.param_distributions = parameters[kernel]

        train_results, tests_results = train_and_test(X_train, y_train, X_test, y_test)

        result[kernel] = (tests_results, train_results)

    result = sorted(result.items(), key=lambda kv: kv[1])

    if print_result:
        print(np.array(result), f"Len result: {len(result)}", f"Time: {time.time() - start_time} seconds", sep='\n\n')

    return result


if __name__ == '__main__':
    main()
