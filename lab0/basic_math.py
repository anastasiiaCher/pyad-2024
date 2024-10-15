import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    m, n = len(matrix_a), len(matrix_a[0])
    n1, p = len(matrix_b), len(matrix_b[0])

    if n != n1:
        raise ValueError

    matrix_c = [[0 for _ in range(m)] for _ in range(p)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return matrix_c


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a_1 = list(map(float, str(a_1).split()))
    a_2 = list(map(float, str(a_2).split()))

    def f(root, coeffs):
        return coeffs[0] * root ** 2 + coeffs[1] * root + coeffs[2]

    def extremum(coeffs):
        x = 0
        a, b, c = coeffs
        if a == 0:
            return None, None
        else:
            x = -b / (2 * a)
        return x, f(x, coeffs)

    x1, y1 = extremum(a_1)
    x2, y2 = extremum(a_2)

    # result_1 = sc.optimize.minimize_scalar(f, args=a_1)  тоже рабочая, но кидает RuntimeWarning
    # result_2 = sc.optimize.minimize_scalar(f, args=a_2)
    #
    # x1, f_x1 = result_1.x, result_1.fun
    # x2, f_x2 = result_2.x, result_2.fun

    a, b, c = [a_1[i] - a_2[i] for i in range(3)]

    if a == 0:
        if b == 0:
            if c == 0:
                return None
            return []
        x = -c / b
        return [(x, f(x, [a, b, c]))]

    det = b ** 2 - 4 * a * c
    if det < 0:
        return []
    x1 = (-b + det ** 0.5) / 2 * a
    x2 = (-b - det ** 0.5) / 2 * a
    if x1 == x2:
        return [x1, f(x1, a_1)]
    else:
        return [(x1, f(x1, a_1)), (x2, f(x2, a_1))]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = sum(x) / n

    m3 = sum((xi - mean) ** 3 for xi in x) / n

    d = sum((xi - mean) ** 2 for xi in x) / n
    st_dev = np.sqrt(d)

    a3 = m3 / (st_dev ** 3)
    return round(a3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = sum(x) / n

    m4 = sum((xi - mean) ** 4 for xi in x) / n

    d = sum((xi - mean) ** 2 for xi in x) / n
    st_dev = np.sqrt(d)

    e4 = m4 / (st_dev ** 4) - 3
    return round(e4, 2)
