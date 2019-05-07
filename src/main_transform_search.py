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
C = 3
gamma = 0.02
test_size = 0.33
split_random_state = 1
model = svm.SVC(kernel='rbf', gamma=gamma, C=C)

# Выбор классов
classes = '0, 1, 4, 5, 6'
# classes = '0, 1, 4, 5, 7'
# classes = '0, 1, 4, 6, 7'
# classes = '0, 1, 4, 7, 9'
# classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

file_name_data_set = '../data/data10mov_no_abs.mat'

# qt = QuantileTransformer()

# search_random_state = 10
# svc = svm.SVC()
# parameters = {
#     'C': np.arange(0.5, 5.1, 0.5),
#     'gamma': np.arange(0.01, 0.5, 0.01),
#     # 'degree': np.arange(1, 5),
#     # 'coef0': np.arange(1, 5)
# }
#
# clf = RandomizedSearchCV(svc, parameters, cv=5, iid=False,
#                          random_state=search_random_state)

# test global variable
kernel_list = ('rbf',)
# kernel_list = ('sigmoid', 'precomputed')
# kernel_list = ('linear', 'poly', 'rbf', 'sigmoid')
# kernel_list = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')

algorithms = [
    ('Standard Scaler', 'Data after standard scaling', StandardScaler()),
    ('Min-Max Scaler', 'Data after min-max scaling', MinMaxScaler()),
    ('Max-Abs Scaler', 'Data after max-abs scaling', MaxAbsScaler()),
    ('Robust Scaler', 'Data after robust scaling', RobustScaler(quantile_range=(25, 75))),
    ('Normalizer', 'Data after sample-wise L2 normalizing', Normalizer()),
    ('Yeo-Johnson Power Transformer', 'Data after power transformation (Yeo-Johnson)', PowerTransformer(method='yeo-johnson')),
    # Преобразование Box-Cox может применяться только к строго положительным данным
    # ('Box-Cox Power Transformer', 'Data after power transformation (Box-Cox)', PowerTransformer(method='box-cox')),
    # Cumulative distribution function - CDF
    ('Quantile Transformer (Normal)', 'Data after quantile transformation (gaussian cdf)', QuantileTransformer(output_distribution='normal')),
    ('Quantile Transformer (Uniform)', 'Data after quantile transformation (uniform cdf)', QuantileTransformer(output_distribution='uniform'))
]


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
    # clf.fit(X_train, y_train)
    # return clf.score(X_train, y_train), clf.score(X_test, y_test)
    model.fit(X_train, y_train)
    return model.score(X_train, y_train), model.score(X_test, y_test)


def main():
    emg_data_set = read_mat(os.path.abspath(file_name_data_set))

    classes_list = list(set(map(int, classes.split(","))))

    X, y = [], []
    for index, value in enumerate(emg_data_set.iloc[classes_list]):
        X.extend(value)
        y.extend([index] * len(value))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state)

    new_X_train, new_X_test = [], []

    for kernel in kernel_list:
        # svc.kernel = kernel
        model.kernel = kernel

        print(f'---- Kernel: {kernel}')

        result = dict()

        for name, title, algorithm in algorithms:
            start_time = time.time()

            if kernel == 'precomputed':
                new_X_train, new_X_test = convert_to_gram_matrix(X_train, X_test)

            new_X_train, new_X_test = preprocessing_data(algorithm, X_train, X_test)

            train_results, tests_results = train_and_test(new_X_train, y_train, new_X_test, y_test)

            result[name] = (train_results, tests_results, time.time() - start_time)

            print(f"[{name}] {title}")
            # print(f"Train result: {train_results:.2%}")
            # print(f"Verification result: {tests_results:.3%}")

        result = sorted(result.items(), key=lambda kv: kv[1])

        print(np.array(result))





if __name__ == '__main__':
    main()
