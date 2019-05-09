import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from numpy.core import multiarray

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline  # make_pipeline более удобный в использовании
from sklearn.pipeline import make_pipeline

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
# C = 3
# gamma = 0.02
test_size = 0.33
split_random_state = 1
# model = svm.SVC(kernel='rbf', gamma=gamma, C=C)

# Выбор классов
# classes = '0, 1, 4, 5, 6'
# classes = '0, 1, 4, 5, 7'
# classes = '0, 1, 4, 6, 7'
# classes = '0, 1, 4, 7, 9'
classes = '0, 1, 4, 5, 6, 7'
# classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

file_name_data_set = '../data/data10mov_no_abs.mat'

pipeline = make_pipeline(Normalizer(), SVC(kernel='poly'))

parameters = {
    'svc__C': np.arange(1, 5.1, 1),
    'svc__gamma': np.arange(0.01, 0.1, 0.01),
    'svc__degree': np.arange(1, 5),
    'svc__coef0': np.arange(1, 5)
}

search_random_state = 10
clf = RandomizedSearchCV(pipeline, parameters, cv=5, iid=False,
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


def train_and_test(X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return clf.score(X_train, y_train), clf.score(X_test, y_test)


def main():
    emg_data_set = read_mat(os.path.abspath(file_name_data_set))

    classes_list = list(set(map(int, classes.split(","))))

    X, y = [], []
    for index, value in enumerate(emg_data_set.iloc[classes_list]):
        X.extend(value)
        y.extend([index] * len(value))

    start_time = time.time()  # time begin (optional)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state)

    train_results, tests_results = train_and_test(X_train, y_train, X_test, y_test)

    print(f"Time: {time.time() - start_time} seconds")  # time end (optional)

    print(f"Train result: {train_results:.2%}")
    print(f"Verification result: {tests_results:.3%}")


if __name__ == '__main__':
    main()
