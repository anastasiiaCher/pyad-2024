import math

import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    rows_a, columns_a, rows_b, columns_b = len(matrix_a), len(matrix_a[0]), len(matrix_b), len(matrix_b[0])

    result = []

    if columns_a != rows_b:
        raise ValueError()

    result = []
    for i in range(rows_a):
        result.append([0] * columns_b)

    for i in range(rows_a):
        for k in range(columns_b):
            for j in range(rows_b):
                result[i][k] += matrix_a[i][j] * matrix_b[j][k]

    return result


def functions(a_1, a_2):
    c1 = list(map(float, a_1.split()))
    c2 = list(map(float, a_2.split()))

    if a_1 == a_2:
        return None

    c = [0, 0, 0]
    result = list()

    for i in range(3):
        c[i] = c1[i] - c2[i]

    if c[0] == 0 and c[1] == 0:
        return result
    elif c[0] == 0 and c[2] != 0:
        result.append(-c[2]/c[1])
    else:
        result = find_roots(c)

    return [(x, quadratic(c1, x)) for x in result]


def find_roots(c):
    d = c[1] ** 2 - 4 * c[0] * c[2]
    r = list()
    if d == 0:
        r.append(-math.sqrt(c[2])/math.sqrt(c[0]))
    elif d > 0:
        r.append((-c[1] + math.sqrt(d)) / (2 * c[0]))
        r.append((-c[1] - math.sqrt(d)) / (2 * c[0]))

    return r

def quadratic(c, x):
    return c[0] * x ** 2 + c[1] * x + c[2]


def skew(x):
    disp = calc_moment(x, 2)
    m3 = calc_moment(x, 3)

    return round(m3 / (disp ** (3 / 2)), 2)

def calc_moment(x, n):
    avg = sum(x) / len(x)

    summ = 0

    for i in x:
        summ += (i - avg) ** n

    return summ / len(x)


def kurtosis(x):
    disp = calc_moment(x, 2)
    m4 = calc_moment(x, 4)

    return round((m4 / (disp ** (4 / 2))) - 3, 2)