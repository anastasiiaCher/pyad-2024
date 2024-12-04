from math import sqrt, pow

import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    if len(matrix_a[0]) != len(matrix_b) or len(matrix_a) != len(matrix_b[0]):
        raise ValueError

    result = [[0] * len(matrix_a) for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_a)):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def find_roots(a: float, b: float, c: float) -> list[float]:
    if a == 0:
        if b == 0:
            return []

        return [-c / b]
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return []
    
    if discriminant == 0:
        root = -b / (2 * a)
        return [root]
    
    root1 = (-b + sqrt(discriminant)) / (2 * a)
    root2 = (-b - sqrt(discriminant)) / (2 * a)

    return [root1, root2]


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    coeffs1 = list(map(float, a_1.split()))
    coeffs2 = list(map(float, a_2.split()))

    if coeffs1 == coeffs2:
        return None
    
    coeffs_diff = [x1 - x2 for x1, x2 in zip(coeffs1, coeffs2)]

    roots = find_roots(*coeffs_diff)
    result = []

    for root in roots:
        value = 0
        for index, coeff in enumerate(coeffs2[::-1]):
            value += coeff * root**index

        result.append((root, value))

    return result


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """

    mean = sum(x) / len(x)
    
    variance = sum(pow(n - mean, 2) for n in x) / len(x)
    std_dev = pow(variance, 0.5)
    
    skew = (sum(pow(n - mean, 3) for n in x) / len(x)) / pow(std_dev, 3)
    
    return round(skew, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """

    mean = sum(x) / len(x)
    
    variance = sum(pow(n - mean, 2) for n in x) / len(x)
    std_dev = pow(variance, 0.5)
    
    kurtosis = (sum(pow(n - mean, 4) for n in x) / len(x)) / pow(std_dev, 4)
    
    excess_kurtosis = kurtosis - 3
    
    return round(excess_kurtosis, 2)
