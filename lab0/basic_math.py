import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    a_rows, a_cols = len(matrix_a), len(matrix_a[0])
    b_rows, b_cols = len(matrix_b), len(matrix_b[0])
    if a_cols != b_rows:
        raise ValueError("Число столбцов должны совпадать")
    result = [[0 for _ in range(b_cols)] for _ in range(a_rows)]

    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    if a11 == a21 and a12 == a22 and a13 == a23:
        return None

    F = lambda x: a11 * x**2 + a12 * x + a13
    P = lambda x: a21 * x**2 + a22 * x + a23
    extremum_F = sc.optimize.fmin(lambda x: F(x), 0, disp=False)[0]
    extremum_P = sc.optimize.fmin(lambda x: P(x), 0, disp=False)[0]

    A = a11 - a21
    B = a12 - a22
    C = a13 - a23
    D = B*B - 4 * A * C
    if A == 0:
        if B == 0:
            return [] if C != 0 else None
        else:
            x = -C / B
            return [(round(x, 2), round(F(x), 2))]
    else:
        if D > 0:
            x1 = (-B + np.sqrt(D)) / (2 * A)
            x2 = (-B - np.sqrt(D)) / (2 * A)
            return [(round(x1, 2), round(F(x1), 2)), (round(x2, 2), round(F(x2), 2))]
        elif D == 0:
            x = -B / (2 * A)
            return [(round(x, 2), round(F(x), 2))]
        else:
            return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = np.mean(x)
    m3 = np.sum((xi - mean) ** 3 for xi in x) / n
    std_dev = np.std(x)
    if std_dev == 0:
        return float('nan')
    return round(m3 / std_dev**3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = np.mean(x)
    m4 = np.sum((xi - mean) ** 4 for xi in x) / n
    std_dev = np.std(x)
    if std_dev == 0:
        return float('nan')
    return round(m4 / std_dev**4 - 3, 2)
