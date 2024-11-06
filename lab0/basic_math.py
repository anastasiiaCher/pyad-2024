import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    shape_a, shape_b = (
        (len(matrix_a), len(matrix_a[0])),
        (len(matrix_b), len(matrix_b[0])),
    )

    if shape_a[1] != shape_b[0]:
        raise ValueError(f"invalid matrix shape: {shape_a=}, {shape_b=}")

    matrix_c = [
        [
            sum(matrix_a[i][k] * matrix_b[k][j] for k in range(shape_a[1]))
            for j in range(shape_b[1])
        ]
        for i in range(shape_a[0])
    ]

    return matrix_c


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coefs_1 = np.array(list(map(float, a_1.split())))
    coefs_2 = np.array(list(map(float, a_2.split())))

    if np.array_equal(coefs_1, coefs_2):
        return None

    F = lambda x: np.polyval(coefs_1, x)

    A, B, C = coefs_1 - coefs_2

    if A == 0:
        if B == 0:
            return [] if C != 0 else None
        root = -C / B
        return [(root, F(root))]

    D = B ** 2 - 4 * A * C

    if D < 0:
        return []

    if D == 0:
        root = -B / (2 * A)
        return [(root, F(root))]

    root1 = (-B + np.sqrt(D)) / (2 * A)
    root2 = (-B - np.sqrt(D)) / (2 * A)
    return sorted([(root1, F(root1)), (root2, F(root2))])


def n_momentum(x, n):
    return np.mean((x - np.mean(x)) ** n)


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    return round(n_momentum(x, 3) / n_momentum(x, 2) ** (3/2), 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    return round(n_momentum(x, 4) / n_momentum(x, 2) ** 2 - 3, 2)
