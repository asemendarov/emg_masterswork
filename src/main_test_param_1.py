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
left_cup, right_cup, cup_flag = 100, 70, True # Усечение сигнала слева и справа
classes, classes_flag = '0, 1, 4, 6, 7', True # Выбор классов
dataFileName = '../data/data10mov_no_abs.mat'
model = svm.SVC(kernel='rbf', gamma=gamma, C=C)  # kernel = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
qt = QuantileTransformer()

# test global variable
c_list=np.arange(3.5, 5.1, 0.5)
gamma_list=np.arange(0.01, 0.03, 0.01)
random_state_list = np.arange(45, 46, 1)


def read_mat(fileName):
    mat = sio.loadmat(fileName)
    data = pd.Series([value[0] for value in mat['data']], [
        'Кисть вверх',                  #0
        'Кисть вниз',                   #1
        'Сжатие всех пальцев',          #2
        'Сжатие указ пальца',           #3
        'Сжатие среднего пальца',       #4
        'Сжатие безымянного пальца',    #5
        'Щелчок большого с средним',    #6
        'Разжимание всех пальцев',      #7
        'Поворот кисти влево',          #8
        'Поворот кисти вправо'          #9
    ])
    return data


def train (X_train, y_train, quantile_transform = True):
    if quantile_transform:
        X_train = qt.fit_transform(X_train)
    model.fit(X_train, y_train)
    return model.score(X_train, y_train)


def test (X_test, y_test, quantile_transform = True):
    if quantile_transform:
        X_test = qt.transform(X_test)
    return model.score(X_test, y_test)


if __name__ == '__main__':
    data = read_mat(os.path.abspath(dataFileName))

    classes = list(map(int, classes.split(",")))

    X, y = [], []
    for index, value in enumerate(data):
        if not (classes_flag and index in classes): # optional
            continue

        X.extend(value)
        y.extend([index] * len(value))

    if cup_flag: # optional
        for index, value in enumerate(X):
            X[index] = value[left_cup:-right_cup]


    import my_time as time
    time.start()

    result = dict()
    for rsl in random_state_list:
        for cl in c_list:
            for gl in gamma_list:
                model = svm.SVC(kernel='rbf', gamma=gl, C=cl)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=rsl)

                train_results = train(X_train, y_train)
                tests_results = test(X_test, y_test)

                result[(rsl, cl, gl)] = (tests_results, train_results)

    result = np.array(sorted(result.items(), key=lambda kv: kv[1]))

    print(result, f"Len result: {len(result)}", f"Time: {time.stop()} seconds", sep='\n\n')

