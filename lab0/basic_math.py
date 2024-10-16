import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы.")

    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result


def functions(a_1, a_2):
    bound = (-1000, 1000)
    a_11, a_12, a_13 = map(int, a_1.split())
    a_21, a_22, a_23 = map(int, a_2.split())

    if a_11 > 0:
        res_1_x = sc.optimize.minimize_scalar(
            lambda x: a_11 * x ** 2 + a_12 * x + a_13,
            bounds=bound,
            method='bounded'
        ).x
    else:
        res_1_x = sc.optimize.minimize_scalar(
            lambda x: -a_11 * x ** 2 - a_12 * x - a_13,
            bounds=bound,
            method='bounded'
        ).x
    res_1_y = a_11 * res_1_x ** 2 + a_12 * res_1_x + a_13  # экстремум первой функции
    if a_21 > 0:
        res_2_x = sc.optimize.minimize_scalar(
            lambda x: a_21 * x ** 2 + a_22 * x + a_23,
            bounds=bound,
            method='bounded'
        ).x
    else:
        res_2_x = sc.optimize.minimize_scalar(
            lambda x: -a_21 * x ** 2 - a_22 * x - a_23,
            bounds=bound,
            method='bounded'
        ).x
    res_2_y = a_21 * res_2_x ** 2 + a_22 * res_2_x + a_23  # экстремум второй функции

    a, b, c = a_11 - a_21, a_12 - a_22, a_13 - a_23
    if a == 0 and b == 0:
        if c == 0:
            return None
        else:
            return []
    elif a == 0:
        ans_x = -c / b
        ans_y = a_11 * ans_x ** 2 + a_12 * ans_x + a_13
        return [(ans_x, ans_y)]
    else:
        d = b ** 2 - 4 * a * c
        if d < 0:
            return []
        ans = []
        ans_1_x = (-b + np.sqrt(d)) / (2 * a)
        ans_1_y = a_11 * ans_1_x ** 2 + a_12 * ans_1_x + a_13
        ans.append((ans_1_x, ans_1_y))
        if d != 0:
            ans_2_x = (-b - np.sqrt(d)) / (2 * a)
            ans_2_y = a_11 * ans_2_x ** 2 + a_12 * ans_2_x + a_13
            ans.append((ans_2_x, ans_2_y))
        return ans


def skew(x):
    n = len(x)
    x_sr = sum(x) / n
    sig = np.sqrt(sum([(x_i - x_sr) ** 2 for x_i in x]) / n)
    m3 = sum([(x_i - x_sr) ** 3 for x_i in x]) / n
    a3 = m3 / (sig ** 3)
    return np.round(a3, 2)


def kurtosis(x):
    n = len(x)
    x_sr = sum(x) / n
    sig = np.sqrt(sum([(x_i - x_sr) ** 2 for x_i in x]) / n)
    m4 = sum([(x_i - x_sr) ** 4 for x_i in x]) / n
    e4 = m4 / (sig ** 4) - 3
    return np.round(e4, 2)


functions("0 0 0", "1 2 3")
