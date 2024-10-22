import numpy as np
import math


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    num_rows_a = len(matrix_a)
    num_cols_a = len(matrix_a[0])
    num_cols_b = len(matrix_b[0])
    if num_cols_a != len(matrix_b):
        raise ValueError()
    ans = [[0] * num_cols_b for _ in range(num_rows_a)]
    for i in range(num_rows_a):
        for j in range(num_cols_b):
            for k in range(num_cols_a):
                ans[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return ans


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    if a_1 == a_2:
        return None
    a1, b1, c1 = map(int, a_1.split())
    a2, b2, c2 = map(int, a_2.split())
    a3, b3, c3 = (a1 - a2), (b1 - b2), (c1 - c2)
    #   def f1(x):
    #       return a1 * x ** 2 + b1 * x + c1
    #   def f2(x):
    #       return a2 * x ** 2 + b2 * x + c2
    #   result_f1 = sc.optimize.minimize_scalar(f1)
    #   result_f2 = sc.optimize.minimize_scalar(f2)
    if a3 == 0:
        if b3 == 0:
            return []
        x = -c3 / b3
        return [(x, a3 * x ** 2 + b3 * x + c3)]
    dis = b3 ** 2 - 4 * a3 * c3
    if dis < 0:
        return None
    elif dis == 0:
        x = -(b3 / (2 * a3))
        return [int(x), int(a3 * x ** 2 + b3 * x + c3)]
    sqrt_dis = int(math.sqrt(dis))
    x_plus = int((-b3 + sqrt_dis) / (2 * a3))
    x_minus = int((-b3 - sqrt_dis) / (2 * a3))
    return [(x_plus, int(a1 * x_plus ** 2 + b1 * x_plus + c1)), (x_minus, int(a2 * x_minus ** 2 + b2 * x_minus + c2))]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    #   Просто зашел на гитхаб, посмотрел как работает scipy.stats_py.py
    #   https://github.com/scipy/scipy/blob/main/scipy/stats/_stats_py.py
    #   после чего совместил нужное из трёх функций
    a = np.asarray(x)
    if a.ndim == 0:
        a = np.reshape(a, (-1,))
    mean = np.mean(a, 0, keepdims=True)
    m2 = np.mean((a - mean) ** 2, axis=0)
    m3 = np.mean((a - mean) ** 3, axis=0)
    with np.errstate(all='ignore'):
        eps = np.finfo(m2.dtype).eps
        zero = m2 <= (eps * mean.squeeze(axis=0)) ** 2
        vals = np.where(zero, np.nan, (m3 / m2 ** 1.5))
    return np.round(vals, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    #   Также зашел на гит scipy и переработал их три функции в одну без доп моментов
    a = np.asarray(x)
    if a.ndim == 0:
        a = np.reshape(a, (-1,))
    mean = np.mean(a, axis=0, keepdims=True)
    m2 = np.mean((a - mean) ** 2, axis=0)
    m4 = np.mean((a - mean) ** 4, axis=0)
    with np.errstate(all='ignore'):
        val = (m4 / m2 ** 2) - 3
    return np.round(val, 2)
