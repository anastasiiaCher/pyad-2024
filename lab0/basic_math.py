import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    rows_A = len(matrix_a)
    cols_A = len(matrix_a[0])
    rows_B = len(matrix_b)
    cols_B = len(matrix_b[0])

    if cols_A != rows_B:
        raise ValueError()

    res = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                res[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return res


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a_1 = list(map(float, a_1.split()))
    a_2 = list(map(float, a_2.split()))

    a = a_1[0] - a_2[0]
    b = a_1[1] - a_2[1]
    c = a_1[2] - a_2[2]

    if a == 0 and b == 0 and c == 0:
        return None
    if a == 0:
        if b == 0:
            return []
        else:
            x = -c / b
            return [(x, a_1[0] * x ** 2 + a_1[1] * x + a_1[2])]

    D = b ** 2 - 4 * a * c
    if D < 0:
        return []
    elif D == 0:
        x = -b / (2 * a)
        return [(x, a_1[0] * x ** 2 + a_1[1] * x + a_1[2])]
    else:
        x1 = (-b + np.sqrt(D)) / (2 * a)
        x2 = (-b - np.sqrt(D)) / (2 * a)
        return [(x1, a_1[0] * x1 ** 2 + a_1[1] * x1 + a_1[2]), (x2, a_1[0] * x2 ** 2 + a_1[1] * x2 + a_1[2])]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    avg = sum(x) / len(x)
    m3 = sum((i - avg) ** 3 for i in x) / len(x)

    variance = sum((i - avg) ** 2 for i in x) / len(x)
    sigma = variance ** 0.5

    return round(m3 / sigma ** 3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    avg = sum(x) / len(x)
    m4 = sum((i - avg) ** 4 for i in x) / len(x)
    variance = sum((i - avg) ** 2 for i in x) / len(x)
    sigma = variance ** 0.5
    return round((m4 / sigma ** 4) - 3, 2)
