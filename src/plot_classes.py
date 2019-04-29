import time
import numpy as np
import matplotlib.pyplot as plt
import src.main_classes_search as cl_search

plt.style.use('ggplot')

# global variable
begin_class = 9
end_class = 10

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax1, ax2 = ax.flatten()
ax1.set_title('Train Result')
ax2.set_title('Test Result')
# ax.set_xlabel('')
# ax.set_ylabel('')


def plot(train_test_result):
    test_result_list, train_result_list = zip(*dict(train_test_result).values())

    ax1.plot(train_result_list, 'ro-')
    ax2.plot(test_result_list, 'ro-')

    plt.grid()
    plt.show()


def main():
    cl_search.classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

    start_time = time.time()

    result = list()

    for combinations_len in range(begin_class, end_class + 1):
        print(f'Start ({combinations_len}/{end_class})')

        cl_search.combinations_len = combinations_len
        out_search = cl_search.main(print_result=False)
        result.append(out_search[-1])

        print(f'Stop ({combinations_len}/{end_class}). Time: {time.time() - start_time} seconds')

    # result = sorted(result, key=lambda kv: kv[1])

    print('\n', np.array(result))

    plot(result)


if __name__ == '__main__':
    main()
