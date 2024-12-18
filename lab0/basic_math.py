import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) == len(matrix_b):
        c = []
        for _ in range(len(matrix_a)):
            c += [[0]*len(matrix_b[0])]
        for i in range(len(matrix_a)):
            for j in range(len(matrix_b[0])):
                for f in range(len(matrix_b)):
                    c[i][j] += matrix_a[i][f] * matrix_b[f][j]
        return c
    else:
        raise ValueError ("Матрицы невозможно перемножить")
    pass


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    if a_1 == a_2:
        return None
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())
    def F(x):
      return a11 * x ** 2 + a12 * x + a13
    if a11 != 0:
        x_extr_1 = (-a12) / (2 * a11)
    if a21 != 0:
        x_extr_2 = (-a22) / (2 * a21)
    res = []
    a = a11 - a21
    b = a12 - a22
    c = a13 - a23
    d = b**2 - 4*a*c
    if a == 0:
        if b != 0:
            x = -c / b
            res.append((x, F(x)))
    elif d > 0:
        x1 = (-b + (d)**0.5) / (2*a)
        x2 = (-b - (d)**0.5) / (2*a)
        res.append((x1, F(x1)))
        res.append((x2, F(x2)))
    return res
    pass


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m3 = 0
    d2 = 0
    n = len(x)
    e = sum(x) / n
    for i in x:
        m3 += (i - e) ** 3
        d2 += (i - e) ** 2
    m3 = m3 / n
    d2 = d2 / n
    A = m3 / (d2 ** (3 / 2))
    return round(A, 2)
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m4 = 0
    d2 = 0
    n = len(x)
    e = sum(x) / n
    for i in x:
        m4 += (i - e) ** 4
        d2 += (i - e) ** 2
    m4 = m4 / n
    d2 = d2 / n
    E = (m4 / (d2**2))-3
    return round(E, 2)
    pass
