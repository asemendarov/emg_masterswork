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

# global variable
C = 4
gamma = 0.02
test_size = 0.33
random_state = 1
# left_cup, right_cup, cup_flag = 100, 70, True  # Усечение сигнала слева и справа
classes = '0, 1, 4, 6, 7'  # Выбор классов
# classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'
file_name_data_set = '../data/data10mov_no_abs.mat'
model = svm.SVC(kernel='rbf', gamma=gamma, C=C)  # kernel = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
qt = QuantileTransformer()

# test global variable
range_left_cup = range(49, 50)
range_right_cup = range(44, 54)
cup_flag = True


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


def main(print_result=True):
    emg_data_set = read_mat(os.path.abspath(file_name_data_set))

    classes_list = list(set(map(int, classes.split(","))))

    start_time = time.time()

    result = dict()

    for left_cup in range_left_cup:
        for right_cup in range_right_cup:

            X, y = [], []
            for index, value in enumerate(emg_data_set.iloc[classes_list]):
                X.extend(value)
                y.extend([index] * len(value))

            if cup_flag:  # optional
                for index, value in enumerate(X):
                    X[index] = value[left_cup:-right_cup]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)

            train_results = train(X_train, y_train)
            tests_results = test(X_test, y_test)

            result[(left_cup, right_cup)] = (tests_results, train_results)

    result = sorted(result.items(), key=lambda kv: kv[1])

    if print_result:
        print(np.array(result), f"Len result: {len(result)}", f"Time: {time.time() - start_time} seconds", sep='\n\n')

    return result


'''
    range_left_cup = 49
    range_right_cup = 44..54
    Len result: 10
    Time: 6.125214576721191 seconds
'''
if __name__ == '__main__':
    main()
