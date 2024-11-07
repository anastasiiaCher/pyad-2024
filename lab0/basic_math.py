import numpy as np
import scipy


def matrix_multiplication(matrix_a, matrix_b):
    if matrix_a.shape[1] != matrix_b.shape[0]:
        return f"Число столбцов в первой матрице не равно числу строк во второй матрице"

    zero_matrix = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
    flat_matrix1 = matrix_a.flatten()
    flat_matrix2 = matrix_b.flatten()

    
    for index in range(len(zero_matrix.flatten())):
        row = index // matrix_b.shape[1]
        col = index % matrix_b.shape[1]
        zero_matrix[row, col] = sum(
            flat_matrix1[row * matrix_a.shape[1] + k] * flat_matrix2[k * matrix_b.shape[1] + col] for k in
            range(matrix_a.shape[1]))

    return zero_matrix


def functions(a_1, a_2):
    def quad_func(x, a, b, c):
        return a * x ** 2 + b * x + c

    range_x = np.asarray([-100, 100])
    ex_1 = scipy.optimize.minimize_scalar(quad_func, args=tuple(a_1))
    ex_2 = scipy.optimize.minimize_scalar(quad_func, args=tuple(a_2))

    roots_1 = scipy.optimize.fsolve(quad_func, args=tuple(a_1), x0=range_x)
    roots_2 = scipy.optimize.fsolve(quad_func, args=tuple(a_2), x0=range_x)

    message = ''

    if a_2[2] == a_1[0] == 0:
        message = 'Квадртаные уравнения вырождены в 0. Решений бесконечно много'
    elif np.all(a_1 == a_2):
        message = 'Уравнения совпадают. Два общих решения'
    elif a_2[2] == a_1[0] == 0:
        message = 'Квадртаные уравнения вырождены в 0. Решений бесконечно много'
    elif np.isin(roots_1, roots_2).any() == True:
        message = 'Есть одно общее решение'
    else:
        message = 'Общих решений нет'

    return f" Значение x экстремума первой функции {float(ex_1['x'])} \n Значение x экстремума второй функции {ex_2['x']} \n Корни первого и второго уравнения {roots_1} {roots_2} \n {message}"


def skew(x):
    assym = scipy.stats.moment(x, moment=3) / (scipy.stats.moment(x, moment=2) ** 0.5) ** 3
    return float(assym)


def kurtosis(x):
    exxec = scipy.stats.moment(x, moment=4) / (scipy.stats.moment(x, moment=2) ** 0.5) ** 4 - 3
    return float(exxec)

