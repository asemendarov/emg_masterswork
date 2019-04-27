import time
import numpy as np
import matplotlib.pyplot as plt
import src.main_kernel_search as kernel_search


# global variable
fig, ax = plt.subplots(figsize=(8, 6))


def print_bar(data):
    data = dict(data)

    kernel_list = data.keys()

    index_list = np.arange(len(kernel_list))

    x1 = index_list - 0.2
    x2 = index_list + 0.2

    test_result_list, train_result_list = zip(*data.values())

    ax.bar(x1, test_result_list, width=0.4, label='test_result')
    ax.bar(x2, train_result_list, width=0.4, label='train_result')

    plt.xticks(index_list, kernel_list)

    plt.legend()
    plt.grid(axis='y', which='both')
    plt.show()


def main():
    out_kernel_search = kernel_search.main()
    print_bar(out_kernel_search)


if __name__ == '__main__':
   main()