import os
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.svm as svm
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# global variable
C = 4 # Параметр регуляризации SVM
gamma = 0.02  # Train results: 99.433 / Verification results: 88.123
test_size = 0.33
# random_state = 45
random_state = 1

# Усечение сигнала слева и справа
# left_cup, right_cup, cup_flag = 1, 1, True
left_cup, right_cup, cup_flag = 100, 70, True

# Выбор классов
# classes, classes_flag = '0, 1, 4, 5, 6', True # Train: 100.000 / Verification: 96.183 (cvup = 25) / Verification: 96.947 (cup = [100, 70])
# classes, classes_flag = '0, 2, 4, 6, 7', True # Train: 100.000 / Verification: 96.947 (cvup = 25) / Verification: 98.473 (cup = [100, 70])
# classes, classes_flag = '1, 2, 4, 6, 7', True # Train: 100.000 / Verification: 96.947 (cvup = 25) / Verification: 98.473 (cup = [100, 70])
# classes, classes_flag = '1, 5, 4, 6, 7', True # Train: 100.000 / Verification: 96.947 (cvup = 25) / Verification: 96.183 (cup = [100, 70])
classes, classes_flag = '0, 1, 4, 6, 7', True # Train: 100.000 / Verification: 96.947 (cvup = 25) / Verification: 98.473 (cup = [100, 70], random_state = 45)
# classes, classes_flag = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9', True


dataFileName = 'data/data10mov_no_abs.mat'
newDataFileName = 'data/data10mov_raw.mat'  # Структра хранимых данных отличаетя от dataFileName <<< !!!

# Создаем экземплр SVM и обучаем классификатор
# kernel = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
model = svm.SVC(kernel='rbf', gamma=gamma, C=C)
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=gamma, C=C))

# title for the plots
titles = ('SVC with linear kernel. One-vs-One.',
          'LinearSVC (linear kernel). One-vs-All.',
          'SVC with RBF kernel')

qt = QuantileTransformer()


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

    # import parameters_test as pt
    #     # pt.test2(data, classes_flag, classes)

    # import parameters_test as pt
    # pt.test3(data, classes, 5)

    X, y = [], []
    for index, value in enumerate(data):
        if not (classes_flag and index in classes): # optional
            continue

        X.extend(value)
        y.extend([index] * len(value))

    if cup_flag: # optional
        for index, value in enumerate(X):
            X[index] = value[left_cup:-right_cup]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    train_results = train(X_train, y_train)
    tests_results = test(X_test, y_test)

    print(f"Train result: {train_results:.2%}")
    print(f"Verification result: {tests_results:.3%}")


