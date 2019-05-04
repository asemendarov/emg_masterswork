import time
import numpy as np
import matplotlib.pyplot as plt
import src.main_kernel_search as kernel_search

plt.style.use('ggplot')

# global variable
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax1, ax2 = ax.flatten()
ax1.set_title('С квантильным преобразованием')
ax2.set_title('Без квантильного преобразования')
ax1.set_xlabel('Ядро')
ax1.set_ylabel('Точность классификации')
ax2.set_xlabel('Ядро')
# ax2.set_ylabel('Точность')


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
    kernel_search.classes = '0, 1, 2, 3, 4, 5, 6, 7, 8'

    out_kernel_search = kernel_search.main()
    set_bar(out_kernel_search, ax1)

    # for flag in (True, False):
    #     kernel_search.quantile_transform_flag = flag
    #     out_kernel_search = kernel_search.main()
    #     set_bar(out_kernel_search, ax1 if flag else ax2)

    show_bar()


if __name__ == '__main__':
   main()