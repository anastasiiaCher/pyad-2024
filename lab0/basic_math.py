import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError()
    matrix_c = []
    for i in range(len(matrix_a)):
        matrix_c.append([])
        for k in range(len(matrix_b[0])):
            c = 0
            for j in range(len(matrix_a[0])):
                c += matrix_a[i][j] * matrix_b[j][k]
            matrix_c[i].append(c)
    # put your code here
    return matrix_c


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
        min_x_1 = (-b1) / (2 * a1)
    if a2 != 0:
        min_x_2 = (-b2) / (2 * a2)
    a = a1 - a2
    b = b1 - b2
    c = c1 - c2
    if a != 0:
        d = b**2 - 4 * a * c
        if d < 0:
            return []
        elif d == 0:
            x = (-b + d**0.5) / (2 * a)
            return [(x, x**2 * a1 + x * b1 + c1)]
        else:
            x1 = (-b + d**0.5) / (2 * a)
            x2 = (-b - d**0.5) / (2 * a)
            return [(x1, x1**2 * a1 + x1 * b1 + c1), (x2, x2**2 * a1 + x2 * b1 + c1)]
    elif b != 0:
        x = -c / b
        return [(x, x**2 * a1 + x * b1 + c1)]
    else:
        return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m3 = 0
    x_sr = sum(x) / len(x)
    for i in x:
        m3 += (i - x_sr) ** 3
    m3 /= len(x)

    d = 0
    for i in x:
        d += (i - x_sr) ** 2
    d /= len(x)
    return round(m3 / (d ** (3 / 2)), 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m4 = 0
    x_sr = sum(x) / len(x)
    for i in x:
        m4 += (i - x_sr) ** 4
    m4 /= len(x)

    d = 0
    for i in x:
        d += (i - x_sr) ** 2
    d /= len(x)
    return round(m4 / (d ** (4 / 2)) - 3, 2)
