import numpy as np

from lab0.helpers.functions import find_intersections, quadratic_function


def matrix_multiplication(
    matrix_a: list[list[float | int]],
    matrix_b: list[list[float | int]],
):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if (
            len(matrix_a[0]) != len(matrix_b)
            or len(matrix_a) != len(matrix_b[0])
    ):
        raise ValueError("Не совпадают размерности матриц")

    if (
            (len(matrix_a) == 0 or len(matrix_b) == 0)
            or (len(matrix_a[0]) == 0 or len(matrix_b[0]) == 0)
    ):
        raise ValueError("Одна из матриц пустая")

    result = [[0 for _ in range(len(matrix_a))] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_a)):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1: str, a_2: str):
    """
    Задание: На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо определить, есть ли у функций общие решения.
    """
    coeffs_1 = list(map(float, a_1.split()))
    coeffs_2 = list(map(float, a_2.split()))

    intersections = find_intersections(coeffs_1, coeffs_2)

    if intersections is None:
        return None
    elif len(intersections) == 0:
        return []
    else:
        return [(x, quadratic_function(coeffs_1, x)) for x in intersections]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean_x = np.mean(x)

    m_2 = np.sum((x - mean_x) ** 2) / n
    m_3 = np.sum((x - mean_x) ** 3) / n
    sigma = np.sqrt(m_2)
    result = m_3 / sigma ** 3

    return round(result, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean_x = np.mean(x)

    m_2 = np.sum((x - mean_x) ** 2) / n
    m_4 = np.sum((x - mean_x) ** 4) / n
    sigma = np.sqrt(m_2)
    result = (m_4 / sigma ** 4) - 3

    return round(result, 2)
