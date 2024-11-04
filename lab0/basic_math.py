import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar
from math import sqrt


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Размеры матриц не соответствуют правила умножения")
    result = [[sum(matrix_a[i][n] * matrix_b[n][j] for n in range(len(matrix_b))) for j in range(len(matrix_b[0]))] for
              i in range(len(matrix_a))]
    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a1, b1, c1 = map(float, a_1.split())
    a2, b2, c2 = map(float, a_2.split())

    if a_1 == a_2:
        return None
    if a1 != 0:
        extr_F = (-b1) / (2 * a1)
    if a_2 != 0:
        extr_P = (-b2) / (2 * a2)

    a = a1 - a2
    b = b1 - b2
    c = c1 - c2
    discriminant = b ** 2 - 4 * a * c
    if a == 0:
        if b != 0:
            x = -c / b
            return [(x, a1 * x ** 2 + b1 * x + c1)]
    elif discriminant > 0:
        x1 = round((b*(-1) + sqrt(discriminant)) / (2*a), 2)
        x2 = round((b*(-1) - sqrt(discriminant)) / (2*a), 2)
        return [(x1, a1 * x1 ** 2 + b1 * x1 + c1), (x2, a1 * x2 ** 2 + b1 * x2 + c1)]
    else:
        return None
    return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / len(x)
    m3 = np.sum((x - mean) ** 3) / len(x)
    return round(m3 / np.sqrt(m2) ** 3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / len(x)
    m4 = np.sum((x - mean) ** 4) / len(x)
    return round(m4 / np.sqrt(m2) ** 4 - 3, 2)
