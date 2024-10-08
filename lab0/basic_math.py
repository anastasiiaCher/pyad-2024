import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # put your code here
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])

    cols_b = len(matrix_b[0])

    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Невозможно перемножить матрицы: количество столбцов первой матрицы "
                         "не совпадает с количеством строк второй.")

    # Матрица, заполненная нулями
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # put your code here
    def parse_coefficients(coef_str):
        return list(map(float, coef_str.split()))
    a = parse_coefficients(a_1)
    b = parse_coefficients(a_2)

    if a == b:
        return None

    if a[0] != 0:
        x_extremum1 = -a[1] / (2 * a[0])
        y_extremum1 = a[0] * x_extremum1 ** 2 + a[1] * x_extremum1 + a[2]
        extremum1 = (x_extremum1, y_extremum1)
    else:
        extremum1 = None

    if b[0] != 0:
        x_extremum2 = -b[1] / (2 * b[0])
        y_extremum2 = b[0] * x_extremum2 ** 2 + b[1] * x_extremum2 + b[2]
        extremum2 = (x_extremum2, y_extremum2)
    else:
        extremum2 = None

    A = a[0] - b[0]
    B = a[1] - b[1]
    C = a[2] - b[2]
    if A == 0:
        if B == 0:
            return [] if C != 0 else None
        root = -C / B
        return [(root, a[0] * root ** 2 + a[1] * root + a[2])]
    D = B ** 2 - 4 * A * C

    if D < 0:
        return []
    elif D == 0:
        root = -B / (2 * A)
        return [(root, a[0] * root ** 2 + a[1] * root + a[2])]
    else:
        root1 = (-B + D ** 0.5) / (2 * A)
        root2 = (-B - D ** 0.5) / (2 * A)
        return [(root1, a[0] * root1 ** 2 + a[1] * root1 + a[2]),
                (root2, a[0] * root2 ** 2 + a[1] * root2 + a[2])]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    n = len(x)
    mean_x = sum(x) / n

    m3 = sum((xi - mean_x) ** 3 for xi in x) / n

    variance = sum((xi - mean_x) ** 2 for xi in x) / n
    std_dev = variance ** 0.5

    skewness = m3 / (std_dev ** 3)

    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    n = len(x)
    mean_x = sum(x) / n

    variance = sum((xi - mean_x) ** 2 for xi in x) / n
    std_dev = variance ** 0.5

    m4 = sum((xi - mean_x) ** 4 for xi in x) / n

    E4 = m4 / (std_dev ** 4) - 3

    return round(E4, 2)