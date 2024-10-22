import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError

    matrix = [[None for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            s = 0
            for k in range(len(matrix_a[0])):
                s += matrix_a[i][k] * matrix_b[k][j]
            matrix[i][j] = s
    return matrix


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    def get_function(params):
        def function(x):
            return params[0] * x ** 2 + params[1] * x + params[2]
        return function

    def extremum(arr):
        if arr[0] > 0:
            res = sc.optimize.minimize_scalar(get_function(arr), method="Bounded")
        else:
            arr = [-1 * i for i in arr]
            res = sc.optimize.minimize_scalar(get_function(arr), method="Bounded")
        return res

    a1 = list(map(float, a_1.split()))
    a2 = list(map(float, a_2.split()))
    a3 = [i - j for i, j in zip(a1, a2)]

    if a3 == [0, 0, 0]:
        return None

    roots = np.roots(a3)

    for num in roots:
        if num.imag != 0:
            return []

    y = [get_function(a1)(float(x)) for x in roots]
    res = list((zip(map(float, roots), y)))
    res.reverse()
    return res


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """

    stddev = np.std(x)
    avg = np.mean(x)
    m_3 = sum([(i - avg) ** 3 for i in x]) / len(x)
    return round(m_3 / stddev ** 3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    stddev = np.std(x)
    avg = np.mean(x)
    m_4 = sum([(i - avg) ** 4 for i in x]) / len(x)
    return round(m_4 / stddev ** 4 - 3, 2)
