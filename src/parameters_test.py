from src import main as _main, my_time as time


# import parameters_test as pt
# pt.test1(X, y,
#     random_state_list = np.arange(45, 46, 1),
#     c_list=np.arange(3.5, 5.1, 0.5),
#     gamma_list=np.arange(0.01, 0.3, 0.01)
# )
def test1(X, y, random_state_list, c_list, gamma_list):
    time.start()

    result = dict()
    for rsl in random_state_list:
        for cl in c_list:
            for gl in gamma_list:
                _main.model = _main.svm.SVC(kernel='rbf', gamma=gl, C=cl)

                X_train, X_test, y_train, y_test = _main.train_test_split(
                    X, y, test_size=0.33, random_state=rsl)

                train_results = _main.train(X_train, y_train) * 100
                tests_results = _main.test(X_test, y_test) * 100

                result[(rsl, cl, gl)] = (tests_results, train_results)

    result = _main.np.array(sorted(result.items(), key=lambda kv: kv[1]))
    print(result, "len = %s " % len(result), sep='\n\n')

    time.stop_print()


# import parameters_test as pt
# pt.test2(data, classes_flag, classes)
def test2(data, classes_flag=False, classes=None):
    time.start()

    result = {}
    for left_cup in range(49, 101):
        for right_cup in range(44, 101):

            X, y = [], []
            for index, elem in enumerate(data):
                if not (classes_flag and index in classes):  # optional
                    continue

                X.extend(elem)
                y.extend([index] * len(elem))

            for index, value in enumerate(X):
                X[index] = value[left_cup:-right_cup]

            X_train, X_test, y_train, y_test = _main.train_test_split(
                X, y, test_size=_main.test_size, random_state=_main.random_state)

            train_results = _main.train(X_train, y_train) * 100
            tests_results = _main.test(X_test, y_test) * 100

            result[(left_cup, right_cup)] = (tests_results, train_results)

    result = _main.np.array(sorted(result.items(), key=lambda kv: kv[1]))
    print(result)

    time.stop_print()


# import parameters_test as pt
# pt.test3(data, classes)
def test3(data, classes, combinations_len=5):
    import itertools

    result = dict()
    for classes2 in itertools.combinations(classes, combinations_len):

        X, y = [], []
        for index, value in enumerate(data):
            if not (_main.classes_flag and index in classes2):
                continue

            X.extend(value)
            y.extend([index] * len(value))

        if _main.cup_flag:  # optional
            for index, value in enumerate(X):
                X[index] = value[_main.left_cup:-_main.left_cup]

        X_train, X_test, y_train, y_test = _main.train_test_split(
            X, y, test_size=_main.test_size, random_state=_main.random_state)

        train_results = _main.train(X_train, y_train) * 100
        tests_results = _main.test(X_test, y_test) * 100

        result[tuple(classes2)] = (tests_results, train_results)

    result = _main.np.array(sorted(result.items(), key=lambda kv: kv[1]))
    print(result)