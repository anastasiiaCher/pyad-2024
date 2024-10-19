import numpy as np
import scipy as sc
from collections import Counter

def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError
    c = [list() for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for k in range(len(matrix_b[0])):
            c[i].append(sum([matrix_a[i][j]*matrix_b[j][k] for j in range(len(matrix_b))]))
    return c


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a1_list = list(map(float, a_1.split()))
    a2_list = list(map(float, a_2.split()))
    def f1(x):
        return a1_list[0] * x**2 + a1_list[1] * x + a1_list[2]

    def f2(x):
        return a2_list[0] * x**2 + a2_list[1] * x + a2_list[2]

    def check_solutions(a1, a2):
        if (a1[0] - a2[0] == 0) and (a1[1] - a2[1] == 0) and (a1[2] != a2[2]):
            return False
        else:
            return True
    def system(x):
        return f1(x)-f2(x)

    def get_extr(func, coefs):
        if coefs[0] > 0:
            x = round(sc.optimize.minimize_scalar(func).x, 2)
            y = func(x)
            print(f'Экстремум: ({x},{y})')
    get_extr(f1, a1_list)
    get_extr(f2, a2_list)

    if not check_solutions(a1_list, a2_list):
        return []

    x = np.linspace(-4, 4)
    res = sc.optimize.root(system, x)
    x_roots = np.unique(res.x.round())
    y_roots = f2(x_roots)

    roots = [(round(x_roots[i]), round(y_roots[i])) for i in range(len(x_roots))]
    return roots if len(roots) <= 2 else None


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    counter = Counter(x)
    x_avg = sum(x)/len(x)
    m2 = sum([(i - x_avg)**2 * n_i for (i, n_i) in counter.items()])/len(x)
    m3 = sum([(i - x_avg)**3 * n_i for (i, n_i) in counter.items()])/len(x)

    A3 = round(m3/(m2**(3/2)), 2)
    return A3


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    counter = Counter(x)
    x_avg = sum(x)/len(x)
    m2 = sum([(i - x_avg)**2 * n_i for (i, n_i) in counter.items()])/len(x)
    m4 = sum([(i - x_avg)**4 * n_i for (i, n_i) in counter.items()])/len(x)

    E4 = round(m4/(m2**2)-3, 2)
    return E4
