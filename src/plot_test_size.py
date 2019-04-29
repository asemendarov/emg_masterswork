import time
import numpy as np
import matplotlib.pyplot as plt
import src.main_test_size_search as ts_search

# global variable
test_size_list = np.arange(0.1, 0.51, 0.1)

fig, ax = plt.subplots(figsize=(8, 6))


def plot(data):
    test_result_list, train_result_list = zip(*dict(data).values())

    ax.plot(test_size_list, train_result_list, 'bo-', label='train_result')
    ax.plot(test_size_list, test_result_list, 'go-', label='test_result')

    plt.legend()
    plt.grid()
    plt.show()


def main():

    ts_search.test_size_list = test_size_list

    out_search = ts_search.main()

    plot(out_search)


if __name__ == '__main__':
    main()
