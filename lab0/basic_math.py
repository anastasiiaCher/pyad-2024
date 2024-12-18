import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    num_rows_a = len(matrix_a)
    num_cols_a = len(matrix_a[0])
    num_rows_b = len(matrix_b)
    num_cols_b = len(matrix_b[0])

    if num_cols_a != num_rows_b:
        raise ValueError("Cannot multiply the two matrices. Incorrect dimensions.")


    result = [[0 for _ in range(num_cols_b)] for _ in range(num_rows_a)]


    for i in range(num_rows_a):
        for j in range(num_cols_b):
            for k in range(num_cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    coeffs1 = list(map(float, a_1.strip().split()))
    coeffs2 = list(map(float, a_2.strip().split()))


    p1 = np.poly1d(coeffs1)
    p2 = np.poly1d(coeffs2)


    diff_poly = p1 - p2

    if np.allclose(diff_poly.coeffs, 0):
        return None

    roots = np.roots(diff_poly)
    real_roots = roots[np.isreal(roots)].real

    result = []
    for x in real_roots:
        y = p1(x)
        result.append((x, y))

    return result


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    x_i = sum(x) / len(x)
    sigma = (sum([(num - x_i) ** 2 for num in x]) / len(x)) ** 0.5
    m3 = (sum([(num - x_i) ** 3 for num in x]) / len(x))
    a3 = m3 / sigma ** 3
    return round(a3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x_i = sum(x) / len(x)
    sigma = (sum([(num - x_i) ** 2 for num in x]) / len(x)) ** 0.5
    m4 = (sum([(num - x_i) ** 4 for num in x]) / len(x))
    e4 = m4 / sigma ** 4 - 3
    return round(e4, 2)
