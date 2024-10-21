import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы не могут быть перемножены")

    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    if a_1 == a_2:
        return None

    a1, b1, c1 = map(float, a_1.split())
    a2, b2, c2 = map(float, a_2.split())

    if a1 != 0:
        extremum1 = -b1 / (2 * a1)
    else:
        extremum1 = None

    if a2 != 0:
        extremum2 = -b2 / (2 * a2)
    else:
        extremum2 = None

    a = a1 - a2
    b = b1 - b2
    c = c1 - c2

    if a == 0 and b == 0:
        return []

    if a == 0:
        solution = -c / b
        return [(solution, solution**2 * a1 + solution * b1 + c1)]
    else:
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        elif discriminant == 0:
            solution = -b / (2 * a)
            return [(solution, solution ** 2 * a1 + solution * b1 + c1)]
        else:
            sqrt_disc = discriminant ** 0.5
            solution1 = (-b + sqrt_disc) / (2 * a)
            solution2 = (-b - sqrt_disc) / (2 * a)
            return [(solution1, solution1 ** 2 * a1 + solution1 * b1 + c1), (solution2, solution2 ** 2 * a1 + solution2 * b1 + c1)]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = sum(x) / n
    variance = sum((xi - mean) ** 2 for xi in x) / n
    sigma = variance ** 0.5
    m3 = sum((xi - mean) ** 3 for xi in x) / n
    return round(m3 / sigma ** 3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = sum(x) / n
    variance = sum((xi - mean) ** 2 for xi in x) / n
    sigma = variance ** 0.5
    m4 = sum((xi - mean) ** 4 for xi in x) / n
    return round((m4 / sigma ** 4) - 3, 2)