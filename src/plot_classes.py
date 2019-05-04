import time
import numpy as np
import matplotlib.pyplot as plt
import src.main_classes_search as cl_search

plt.style.use('ggplot')

# global variable
begin_combinations_len = 2
end_combinations_len = 10

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
ax1, ax2 = ax.flatten()
ax1.set_title('Train')
ax2.set_title('Test')
ax2.set_xlabel('Классы')
ax1.set_ylabel('Точность классификации')
ax2.set_ylabel('Точность классификации')


def plot(train_test_result):
    train_test_result = dict(train_test_result)

    classes_list = train_test_result.keys()

    # test_result_list, train_result_list = zip(*train_test_result.values())  # 1
    train_result_list, test_result_list = zip(*train_test_result.values())    # 2

    ax1.plot(train_result_list, 'ro-')
    ax2.plot(test_result_list, 'bo-')

    ax1.set_xticks(range(len(classes_list)))
    ax2.set_xticks(range(len(classes_list)))

    classes_list = [''.join(map(str, classes)) for classes in classes_list]

    ax1.set_xticklabels(classes_list)
    ax2.set_xticklabels(classes_list)

    # ax2.set_xticks(range(len(classes_list)))
    # ax2.set_xticklabels(classes_list)

    plt.show()


def main():
    cl_search.classes = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

    start_time = time.time()

    result = list()

    for combinations_len in range(begin_combinations_len, end_combinations_len + 1):
        print(f'Start ({combinations_len}/{end_combinations_len})')

        cl_search.combinations_len = combinations_len
        out_search = cl_search.main(print_result=False)
        result.append(out_search[-1])

        print(f'Stop ({combinations_len}/{end_combinations_len}). Time: {time.time() - start_time} seconds')

    # result = sorted(result, key=lambda kv: kv[1])

    print('\n', np.array(result))

    plot(result)


if __name__ == '__main__':
    main()
