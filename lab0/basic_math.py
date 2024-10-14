import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы невозможно умножить")

    result = [[0] * len(matrix_b[0]) for i in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result
    pass


def functions(a_1, a_2):
    extr1 = find_extremum(a_1[0], a_1[1], a_1[2])
    extr2 = find_extremum(a_2[0], a_2[1], a_2[2])
    a = a_1[0] - a_2[0]
    b = a_1[1] - a_2[1]
    c = a_1[2] - a_2[2]

    discr = b ** 2 - 4 * a * c
    if a == 0 and b == 0 and c == 0:
        return
    elif a == 0 and b== 0:
        return []
    elif a == 0:
        return [(-c / b)]
    else:
        if discr > 0:
            x1 = (-b + np.sqrt(discr)) / (2 * a)
            x2 = (-b - np.sqrt(discr)) / (2 * a)
            return [x1, x2]
        elif discr == 0:
            return [(-b / (2 * a))]
        else:
            return []
    pass


def skew(x):
    mean = sum(x) / len(x)
    m2 = sum((x1 - mean) ** 2 for x1 in x) / len(x)  # Момент второго порядка (дисперсия)
    m3 = sum((x1 - mean) ** 3 for x1 in x) / len(x)  # Момент третьего порядка
    return m3 / (m2 ** (3 / 2))
    pass


def kurtosis(x):
    mean = sum(x) / len(x)
    m2 = sum((x1 - mean) ** 2 for x1 in x) / len(x)  # Момент второго порядка (дисперсия)
    m4 = sum((x1 - mean) ** 4 for x1 in x) / len(x)  # Момент третьего порядка
    return m4 / (m2 ** 2) - 3
    pass

def find_extremum(a1, a2, a3):
    func = lambda x: a1 * x ** 2 + a2 * x + a3
    res = sc.minimize_scalar(func)
    return res.x, func(res.x)

