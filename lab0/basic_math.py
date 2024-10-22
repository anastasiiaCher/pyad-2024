import numpy as np
from numpy.polynomial.polynomial import Polynomial
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов матрицы A не равно числу строк матрицы B")

    output = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(output)):
        for j in range(len(output[i])):
            for k in range(len(matrix_b)):
                output[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return output


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    def parse_params(params: str):
        return [float(p) for p in params.split()]

    def func(params: list[float]):
        return lambda x: params[0] * x ** 2 + params[1] * x + params[2]

    a1 = parse_params(a_1)
    a2 = parse_params(a_2)
    params_difference = [a1[i] - a2[i] for i in range(len(a1))]

    if params_difference == [0, 0, 0]:
        return None

    params_difference.reverse()
    p = Polynomial(params_difference)
    intersection_x = p.roots()

    real_roots = [x.real for x in intersection_x if x.imag == 0]
    if not real_roots:
        return []

    f1 = func(a1)
    intersection_y = [f1(x) for x in real_roots]

    return list(zip(real_roots, intersection_y))


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    pass
