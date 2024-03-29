import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.svm as svm
import matplotlib.pyplot as plt

import itertools

from multiprocessing import Pool
from functools import partial

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# global variable
# C = 4
# gamma = 0.02
test_size = 0.33
random_state = 1
# model = svm.SVC(kernel='rbf', gamma=gamma, C=C)

file_name_data_set = '../data/data10mov_no_abs.mat'

qt = QuantileTransformer()

# test global variable
combinations_len = 5
search_random_state = 10

# Выбор классов
classes = '0, 1, 4, 6, 7'
# classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

svc = svm.SVC(kernel='rbf')
# svc = svm.SVC(kernel='poly')
parameters = {
    'C': np.arange(1, 5.1, 1),
    'gamma': np.arange(0.01, 0.1, 0.01)
}

# clf = GridSearchCV(svc, parameters, cv=5, iid=False)
clf = RandomizedSearchCV(svc, parameters, cv=5, iid=False,
                         random_state=search_random_state)


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


def quantile_transform(X_train, X_test):
    return qt.fit_transform(X_train), qt.transform(X_test)


def train_and_test(X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return clf.score(X_train, y_train), clf.score(X_test, y_test)


def _X(data):
    return np.vstack(data)


def _y(classes, data_len):
    return np.ravel(np.array(classes * data_len).reshape((data_len, -1)).T)


def combinations_multiprocessing(data, classes_list):
    emg_data_set = data.iloc[list(classes_list)]

    X, y = _X(emg_data_set), _y(classes_list, len(emg_data_set[0]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    X_train, X_test = quantile_transform(X_train, X_test)

    train_results, tests_results = train_and_test(X_train, y_train, X_test, y_test)

    # return classes_list, (tests_results, train_results) # 1
    return classes_list, (train_results, tests_results)   # 2


def main(print_result=True) -> list:
    emg_dataset = read_mat(os.path.abspath(file_name_data_set))

    classes_list = list(set(map(int, classes.split(","))))

    start_time = time.time()

    result = list()

    # Code multiprocessing -start
    pool = Pool(processes=3)
    doubler = partial(combinations_multiprocessing, emg_dataset)

    for out_pool in pool.map(doubler, itertools.combinations(classes_list, combinations_len)):
        result.append(out_pool)
    # Code multiprocessing -end

    result = sorted(result, key=lambda kv: kv[1])

    if print_result:
        print(np.array(result), f"Len result: {len(result)}", f"Time: {time.time() - start_time} seconds", sep='\n\n')

    return result


'''
    
'''
if __name__ == '__main__':
    main()
