import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.svm as svm
import matplotlib.pyplot as plt

import itertools

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# global variable
C = 4
gamma = 0.02
test_size = 0.33
random_state = 1
left_cup, right_cup, cup_flag = 100, 70, False  # Усечение сигнала слева и справа
# classes = '0, 1, 4, 6, 7'  # Выбор классов
fileNameDataSet = '../data/data10mov_no_abs.mat'
model = svm.SVC(kernel='rbf', gamma=gamma, C=C)  # kernel = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
qt = QuantileTransformer()

# test global variable
combinations_len = 5
classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'


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


def train(X_train, y_train, quantile_transform=True):
    if quantile_transform:
        X_train = qt.fit_transform(X_train)
    model.fit(X_train, y_train)
    return model.score(X_train, y_train)


def test(X_test, y_test, quantile_transform=True):
    if quantile_transform:
        X_test = qt.transform(X_test)
    return model.score(X_test, y_test)


def main(return_result=False):
    emg_dataset = read_mat(os.path.abspath(fileNameDataSet))

    classes_list = list(map(int, classes.split(",")))

    start_time = time.time()

    result = dict()

    for new_classes in itertools.combinations(classes_list, combinations_len):

        X, y = [], []
        for index, value in enumerate(emg_dataset):
            if not (classes_flag and index in new_classes):  # optional
                continue

            X.extend(value)
            y.extend([index] * len(value))

        if cup_flag:  # optional
            for index, value in enumerate(X):
                X[index] = value[left_cup:-right_cup]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        train_results = train(X_train, y_train)
        tests_results = test(X_test, y_test)

        result[tuple(new_classes)] = (tests_results, train_results)

    result = np.array(sorted(result.items(), key=lambda kv: kv[1]))

    print(result, f"Len result: {len(result)}", f"Time: {time.time() - start_time} seconds", sep='\n\n')

    print(f"\nWarning! cup_flag = {cup_flag}")

    if return_result:
        return result


'''
    combinations_len = 5
    classes = 0..9
    cup_flag = True
    Len result: 252
    Time: 127.05496525764465 seconds
'''
if __name__ == '__main__':
   main()
