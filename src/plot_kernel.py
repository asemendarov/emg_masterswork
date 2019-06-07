import time
import numpy as np
import matplotlib.pyplot as plt
import src.main_kernel_search as kernel_search

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

plt.style.use('grayscale')
plt.rcParams['figure.facecolor'] = 'white'

# global variable
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax1, ax2 = ax.flatten()
ax1.set_xlabel('Ядро')
ax1.set_ylabel('Точность классификации')
ax2.set_xlabel('Ядро')


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

def set_bar(data, ax):
    data = dict(data)

    kernel_list = data.keys()

    index_list = np.arange(len(kernel_list))

    x1 = index_list - 0.2
    x2 = index_list + 0.2

    test_result_list, train_result_list = zip(*data.values())

    ax.bar(x1, test_result_list, width=0.4, label='Test')
    ax.bar(x2, train_result_list, width=0.4, label='Train')

    ax.set_xticks(range(len(kernel_list)))
    ax.set_xticklabels(kernel_list)


def show_bar():
    ax1.legend()
    plt.show()


def main():
    kernel_search.kernel_list = ('linear', 'poly', 'rbf', 'sigmoid')
    kernel_search.classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'
    # kernel_search.classes = '0, 1, 4, 7, 9'

    # algorithm_name = ('Standard Scaler', 'Min-Max Scaler')
    # algorithm_name = ('Max-Abs Scaler', 'Normalizer')
    # algorithm_name = ('Robust Scaler', 'Yeo-Johnson Power Transformer')
    algorithm_name = ('Uniform Quantile Transformer', 'Normal Quantile Transformer')

    kernel_search.algorithm = algorithms[algorithm_name[0]]
    ax1.set_title(algorithm_name[0])
    out_kernel_search = kernel_search.main()
    set_bar(out_kernel_search, ax1)

    kernel_search.algorithm = algorithms[algorithm_name[1]]
    ax2.set_title(algorithm_name[1])
    out_kernel_search = kernel_search.main()
    set_bar(out_kernel_search, ax2)

    show_bar()


if __name__ == '__main__':
   main()