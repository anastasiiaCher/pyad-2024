import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
     if len(matrix_a[0]) == len(matrix_b):
        length = len(matrix_a)
        result_matrix = [[0 for i in range(length)] for i in range(length)]
        for i in range(len(matrix_a)):
          for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
               result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
        return  result_matrix
    else:
        raise ValueError("Матрицы не подходят для умножения")
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
        return a11*x**2 + a12*x + a13
    def P(x):
        return a21*x**2 + a22*x + a23
    if a11 != 0:
        x_extr_1 = (-a12) / (2 * a11)
        F_ans = F(x_extr_1)
    if a21 != 0:
        x_extr_2 = (-a22) / (2 * a21)
        P_ans = P(x_extr_2)
    d = (a12 - a22)2 - 4*(a11 - a21)*(a13 - a23)
    root = []
    if (a11 - a21) == 0:
        if (a12 - a22) != 0:
            x = -(a13 - a23) /(a12 - a22)
            root.append((x, F(x)))
        return root
    elif d > 0:
        x1 = (-(a12 - a22) + (d)**0.5) / (2*(a11 - a21))
        x2 = (-(a12 - a22) - (d)**0.5) / (2*(a11 - a21))
        root.append((x1, F(x1)))
        root.append((x2, F(x2)))
        return root
    else:
        return None
    pass


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m3, t = 0, 0
    for i in x:
        m3 += (i - sum(x) / len(x))  3
        t += (i - sum(x) / len(x))  2
    A3 = (m3/len(x)) / ((t / len(x))  (3 / 2))
    return round(A3, 2)
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m4, t = 0, 0
    for i in x:
        m4 += (i - sum(x) / len(x))  4
        t += (i - sum(x) / len(x))  2
    e4 = (m4/len(x)) / ((t / len(x)) ** 2) - 3
    return round(e4, 2)
    pass
