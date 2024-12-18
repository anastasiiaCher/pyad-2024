import numpy as np
import scipy as sc

def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Incorrect input")

    result = [[0] * len(matrix_b[0]) for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    a1, b1, c1 = map(float, a_1.split())
    a2, b2, c2 = map(float, a_2.split())

    extremum1 = -b1 / (2 * a1) if a1 != 0 else None
    extremum2 = -b2 / (2 * a2) if a2 != 0 else None

    if a_1 == a_2:
        return None

    a_diff = a1 - a2
    b_diff = b1 - b2
    c_diff = c1 - c2

    if a_diff == 0 and b_diff == 0:
        return []

    if a_diff == 0:
        solution = -c_diff / b_diff
        y_value = a1 * solution ** 2 + b1 * solution + c1
        return [(solution, y_value)]
    else:
        discriminant = b_diff ** 2 - 4 * a_diff * c_diff
        if discriminant < 0:
            return None
        elif discriminant == 0:
            solution = -b_diff / (2 * a_diff)
            y_value = a1 * solution ** 2 + b1 * solution + c1
            return [(solution, y_value)]
        else:
            sqrt_disc = discriminant ** 0.5
            x1 = (-b_diff + sqrt_disc) / (2 * a_diff)
            x2 = (-b_diff - sqrt_disc) / (2 * a_diff)
            return [(x1, a1 * x1 ** 2 + b1 * x1 + c1),
                    (x2, a1 * x2 ** 2 + b1 * x2 + c1)]


def skew(x):
    avg = sum(x) / len(x)
    variance = sum((i - avg) ** 2 for i in x) / len(x)
    o = variance ** 0.5
    m3 = 0
    for i in x:
        m3 += (i - avg) ** 3
    m3 /= len(x)
    return round(m3 / o ** 3, 2)


def kurtosis(x):
    avg = sum(x) / len(x)
    variance = sum((xi - avg) ** 2 for xi in x) / len(x)
    o = variance ** 0.5
    m4 = 0
    for i in x:
        m4 += (i - avg) ** 4
    m4 /= len(x)
    return round((m4 / o ** 4) - 3, 2)
