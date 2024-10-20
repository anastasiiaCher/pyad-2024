import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Ошибка: Матрицы можно переумножить только при условии: "
                         "число столбцов первой матрицы равно числу строк второй матрицы")

    rows_a = len(matrix_a)
    columns_a = len(matrix_a[0])
    columns_b = len(matrix_b[0])

    matrix_c = [[0 for _ in range(columns_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(columns_b):
            for k in range(columns_a):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return matrix_c


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a_1 = list(map(float, a_1.split()))
    a_2 = list(map(float, a_2.split()))

    # Находим точки экстремума функций
    if a_1[0] != 0:
        f_extr = - a_1[1] / (2 * a_1[0])
    if a_2[0] != 0:
        p_extr = - a_2[1] / (2 * a_2[0])

    # Создадим список для общих точек
    res = []

    # Если функции совпадают, то есть решений бесконечно много
    if a_1[0] == a_2[0] and a_1[1] == a_2[1] and a_1[2] == a_2[2]:
        return None

    # Если коэффициенты при x**2 равны
    if a_1[0] == a_2[0] and a_1[1] != a_2[1]:
        x = (a_2[2] - a_1[2]) / (a_1[1] - a_2[1])
        y = a_1[0] * x**2 + a_1[1] * x + a_1[2]
        res.append((x, y))
        return res

    # Обычный случай
    a = a_1[0] - a_2[0]
    b = a_1[1] - a_2[1]
    c = a_1[2] - a_2[2]

    discriminant = b**2 - 4 * a * c
    if discriminant >= 0 and a != 0:
        x1 = (-b + discriminant ** 0.5) / (2 * a)
        x2 = (-b - discriminant ** 0.5) / (2 * a)
        if x1 == x2:
            res.append((x1, a_1[0] * x1**2 + a_1[1] * x1 + a_1[2]))
        else:
            res.append((x1, a_1[0] * x1 ** 2 + a_1[1] * x1 + a_1[2]))
            res.append((x2, a_1[0] * x2 ** 2 + a_1[1] * x2 + a_1[2]))

    return res


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    x_e = sum(x) / len(x)
    m2 = 0
    m3 = 0
    for i in x:
        m2 += (i - x_e) ** 2
        m3 += (i - x_e) ** 3
    m2 /= len(x)
    m3 /= len(x)
    koef_asim = m3 / (m2 ** 0.5) ** 3
    return round(koef_asim, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x_e = sum(x) / len(x)
    m2 = 0
    m4 = 0
    for i in x:
        m2 += (i - x_e) ** 2
        m4 += (i - x_e) ** 4
    m2 /= len(x)
    m4 /= len(x)
    koef_exc = m4 / (m2 ** 0.5) ** 4 - 3
    return round(koef_exc, 2)
