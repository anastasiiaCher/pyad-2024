import math
import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    if len(matrix_b) != len(matrix_a[0]):
        raise ValueError

    matrix_c = np.zeros((len(matrix_a), len(matrix_b[0])), dtype=int)

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_a[0])):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return [list(item) for item in matrix_c]


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    numbers = [int(i) for i in a_1.split(' ')]
    numbers2 = [int(i) for i in a_2.split(' ')]

    a1, b1, c1 = numbers
    a2, b2, c2 = numbers2

    a = a1 - a2
    b = b1 - b2
    c = c1 - c2

    d = b ** 2 - 4 * a * c

    if a == 0 and b == 0 and c == 0:
        return None
    if a == 0 and b == 0:
        return []
    if a == 0:
        x = -c / b
        return [(x, a * x ** 2 + b * x + c)]

    if d < 0:
        return []
    if d == 0:
        x = (-b + math.sqrt(d)) / (2 * a)
        return [(x, a1 * x ** 2 + b1 * x + c1)]
    if d > 0:
        x1 = (-b + math.sqrt(d)) / (2 * a)
        x2 = (-b - math.sqrt(d)) / (2 * a)
        return [(x1, a1 * x1 ** 2 + b1 * x1 + c1),
                (x2, a1 * x2 ** 2 + b1 * x2 + c1)]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    """
    Два варианта решения:
    
    std = np.std(x)

    return round(sc.stats.moment(x, moment=3) / std ** 3, 2)"""

    return round(sum((k - sum(x) / len(x)) ** 3 for k in x) / len(x) / (
                math.sqrt(sum((k - sum(x) / len(x)) ** 2 for k in x) / len(x)) ** 3), 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    """
    Два варианта решения:
    
    std = np.std(x)

    return round(sc.stats.moment(x, moment=4) / std ** 4 - 3, 2)"""

    return round(sum((k - sum(x) / len(x)) ** 4 for k in x) / len(x) / (
            math.sqrt(sum((k - sum(x) / len(x)) ** 2 for k in x) / len(x)) ** 4) - 3, 2)

