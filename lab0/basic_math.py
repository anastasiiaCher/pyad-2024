import numpy as np
from scipy.optimize import minimize_scalar, fmin


def f_1(x):
    return a[0] * x * x + a[1] * x + a[2]


def f_2(x):
    return b[0] * x * x + b[1] * x + b[2]


def moment_n(x, degree):
    summ = 0
    for i in range(len(x)):
        summ += x[i]
    x_avr = summ / len(x)

    summ = 0
    for j in range(len(x)):
        summ += pow((x[j] - x_avr), degree)

    ans = summ / len(x)
    return ans


def matrix_multiplication(matrix_a, matrix_b):
    new_matrix = []
    new_len = min(len(matrix_a), len(matrix_b))
    new_elem = 0
    for j in range(new_len):
        new_matrix.append([])
        for w in range(len(matrix_b[0])):
            for q in range(len(matrix_a[0])):
                new_elem += matrix_a[j][q] * matrix_b[q][w]
            new_matrix[-1].append(new_elem)
            new_elem = 0
    return new_matrix


def functions(a_1, a_2):
    global a, b
    ans = []
    a = list(map(float, a_1.split(' ')))
    b = list(map(float, a_2.split(' ')))

    try:
        res_1 = minimize_scalar(f_1)
    except RuntimeError:
        res_1 = None
    try:
        res_2 = minimize_scalar(f_2)
    except RuntimeError:
        res_2 = None

    coef_a = a[0] - b[0]
    coef_b = a[1] - b[1]
    coef_c = a[2] - b[2]
    discrim = pow(coef_b, 2) - 4 * coef_a * coef_c
    if (a == b):
        pass
    elif (coef_a == 0 and coef_b != 0):
        x = (b[2] - a[2]) / (a[1] - b[1])
        y = f_1(x)
        ans.append((x, y))
    elif discrim > 0:
        x_1 = (-1 * coef_b + pow(discrim, 0.5)) / (2 * coef_a)
        x_2 = (-1 * coef_b - pow(discrim, 0.5)) / (2 * coef_a)
        y_1 = f_1(x_1)
        y_2 = f_1(x_2)
        ans.append((x_1, y_1))
        ans.append((x_2, y_2))
    elif discrim == 0:
        if (coef_a == 0 and coef_b == 0):
            pass
        else:
            x = (-1 * coef_b) / (2 * coef_a)
            y = f_1(x)
            ans.append((x, y))
    return ans


def skew(x):
    ans = (moment_n(x, 3)) / pow(pow(moment_n(x, 2), 0.5), 3)
    return round(ans, 2)


def kurtosis(x):
    ans = ((moment_n(x, 4)) / pow(pow(moment_n(x, 2), 0.5), 4)) - 3
    return round(ans, 2)
