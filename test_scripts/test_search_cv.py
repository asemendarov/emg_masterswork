import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.svm as svm
import matplotlib.pyplot as plt

from numpy.core import multiarray

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# global variable
# C = 4
# gamma = 0.02
test_size = 0.33
split_random_state = 1
# model = svm.SVC(kernel='rbf', gamma=gamma, C=C)

# Выбор классов
# classes = '0, 1, 4, 5, 7'
# classes = '0, 1, 4, 6, 7'
# classes = '0, 1, 4, 7, 9'
classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

file_name_data_set = '../data/data10mov_no_abs.mat'

qt = QuantileTransformer()

search_random_state = 10
svc = svm.SVC(kernel='rbf')
parameters = {
    'C': np.arange(1, 5.1, 1),
    'gamma': np.arange(0.01, 0.06, 0.01)
}

clf1 = GridSearchCV(svc, parameters, cv=5, iid=False)
clf2 = RandomizedSearchCV(svc, parameters, cv=5, iid=False,
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


def train_and_test(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return clf.score(X_train, y_train), clf.score(X_test, y_test)


def main():
    emg_data_set = read_mat(os.path.abspath(file_name_data_set))

    classes_list = list(set(map(int, classes.split(","))))

    X, y = [], []
    for index, value in enumerate(emg_data_set.iloc[classes_list]):
        X.extend(value)
        y.extend([index] * len(value))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state)

    X_train, X_test = quantile_transform(X_train, X_test)

    for clf in (clf1, clf2):
        start_time = time.time()  # time begin (optional)

        train_results, tests_results = train_and_test(clf, X_train, y_train, X_test, y_test)

        print(f"Train result: {train_results:}")
        print(f"Verification result: {tests_results}")

        print(f"Time: {time.time() - start_time} seconds")  # time end (optional)




if __name__ == '__main__':
    main()
