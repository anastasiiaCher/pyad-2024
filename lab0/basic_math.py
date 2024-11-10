import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar


def matrix_multiplication(a, b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if (a.shape[1] != b.shape[0]):
        return "Перемножение матриц невозможно"

    c = np.zeros((a.shape[0], b.shape[1]))

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            summ = 0
            for k in range(c.shape[0]):
                summ += a[i][k] * b[k][j]
            c[i][j] = summ

    return c

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coef_f = np.array(a_1.strip().split(), int)
    coef_p = np.array(a_2.strip().split(), int)

    def func_f(x):
        return coef_f[0]*x**2 + coef_f[1]*x + coef_f[2]

    def func_p(x):
        return coef_p[0]*x**2 + coef_p[1]*x + coef_p[2]

    result_f = minimize_scalar(func_f)
    result_p = minimize_scalar(func_p)

    coeffs = coef_f - coef_p

    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)]

    values = np.zeros(real_roots.shape[0]).astype('int')
    for i in range(values.shape[0]):
        values[i] = coef_f[0] * real_roots[i]**2 + coef_f[1] * real_roots[i] + coef_f[2]

    return list(zip(real_roots, values))


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    if isinstance(x, list):
        x = np.array(x)

    unique, counts = np.unique(x, return_counts=True)
    return moment(x, 3) / (x.std() ** 3)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    if isinstance(x, list):
        x = np.array(x)

    unique, counts = np.unique(x, return_counts=True)
    return moment(x, 4) / (x.std() ** 4) - 3


def moment(arr, n):
  unique, counts = np.unique(arr, return_counts=True)
  return ((unique - arr.mean())**n * counts).sum() / arr.shape[0]
