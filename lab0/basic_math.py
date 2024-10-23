from scipy.optimize import minimize_scalar, root
from collections import Counter
import numpy as np
import math


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Умножение матриц невозможно. Число столбцов первой матрицы должно быть равно числу строк второй матрицы.")
    
    number_of_rows = len(matrix_a)
    number_of_columns = len(matrix_b[0])

    output_matrix = [[0] * number_of_columns for _ in range(number_of_rows)]
    
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            for k in range(len(matrix_b)):
                output_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return output_matrix


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coefs_1 = list(map(float, a_1.split()))
    coefs_2 = list(map(float, a_2.split()))

    def f1(x):
        return coefs_1[0]*x**2 + coefs_1[1]*x + coefs_1[2]

    def f2(x):
        return coefs_2[0]*x**2 + coefs_2[1]*x + coefs_2[2]

    def get_extr(func, coefs):
        if coefs[0] > 0:
            return f'Экстремум в точке: {minimize_scalar(func).x}'

    def system(x):
        return f1(x) - f2(x)

    x = np.linspace(-4, 4, 100)
    result = root(system, x)
    x_roots = np.unique(result.x.round())
    y_roots = f2(x_roots)
    final_result = []
    if result.success:
        for i in range(len(x_roots)):
            final_result.append((round(x_roots[i]), round(y_roots[i])))
            if len(final_result) > 2:
                return None
    return final_result


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    cnt = Counter(x)
    n = len(x)
    mean = sum(x) / n

    m2 = sum([(xi - mean) ** 2 * ni for xi, ni in cnt.items()]) / n
    m3 = sum([(xi - mean) ** 3 * ni for xi, ni in cnt.items()]) / n

    sigma = math.sqrt(m2)
    asym = m3 / (sigma ** 3)

    return round(asym, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    count = Counter(x)
    n = len(x)
    mean_value = sum(x) / n

    m2 = sum([(xi - mean_value) ** 2 * ni for xi, ni in count.items()]) / n
    m4 = sum([(xi - mean_value) ** 4 * ni for xi, ni in count.items()]) / n

    ex = m4 / (m2 ** 2) - 3
    
    return round(ex, 2)
