import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    rows_m1, cols_m1 = len(matrix_a), len(matrix_a[0])
    rows_m2, cols_m2 = len(matrix_b), len(matrix_b[0])
    if cols_m1 != rows_m2:
        raise ValueError
    result = [[0 for _ in range(cols_m2)] for _ in range(rows_m1)]
    
    for i in range(rows_m1):
        for j in range(cols_m2):
            for k in range(cols_m1):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result
  


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    def find_extr(cf):
        def func(x):
            return cf[0] * x ** 2 + cf[1] * x + cf[2]

        result = minimize_scalar(func)
        return round(result.x, 0), func(result.x)

    F_xc = list(map(int, a_1.split()))
    P_xc = list(map(int, a_2.split()))
    extr_1 = find_extr(F_xc)
    extr_2 = find_extr(P_xc)

    if F_xc == P_xc:
        return None

    a = F_xc[0] - P_xc[0]
    b = F_xc[1] - P_xc[1]
    c = F_xc[2] - P_xc[2]

    if a == 0 and b != 0:
        x = -c / b
        return [(x, F_xc[0] * x ** 2 + F_xc[1] * x + F_xc[2])]
    elif a == 0 and b == 0:
        return []

    discr = b ** 2 - 4 * a * c

    if discr > 0:
        x1 = (-b + np.sqrt(discr)) / (2 * a)
        x2 = (-b - np.sqrt(discr)) / (2 * a)
        return [(x1, F_xc[0]*x1**2 + F_xc[1]*x1 + F_xc[2]), (x2, F_xc[0]*x2**2 + F_xc[1]*x2 + F_xc[2])]
    elif discr == 0:
        x = -b / (2 * a)
        return [(x, F_xc[0]*x**2 + F_xc[1]*x + F_xc[2])]

    else:
        return []



def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    def moment(i, mean, sel, n):
        m_sum = 0
        for key, value in sel.items():
            m_sum += (key - mean) ** i * value
        m = m_sum / n
        return m

    n = len(x)
    sel = {}
    for i in x:
        if i not in sel:
            sel[i] = 1
        else:
            sel[i] += 1
    sum = 0

    for key, value in sel.items():
        sum += key * value

    mean = sum / n
    D = moment(2, mean, sel, n)
    m3 = moment(3, mean, sel, n)
    sigma = np.sqrt(D)

    A = round(m3 / (sigma ** 3), 2)
    return A

def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    def moment(i, mean, sel, n):
        m_sum = 0
        for key, value in sel.items():
            m_sum += (key - mean) ** i * value
        m = m_sum / n
        return m

    n = len(x)
    sel = {}
    for i in x:
        if i not in sel:
            sel[i] = 1
        else:
            sel[i] += 1
    sum = 0

    for key, value in sel.items():
        sum += key * value

    mean = sum / n
    D = moment(2, mean, sel, n)
    m4 = moment(4, mean, sel, n)
    sigma = np.sqrt(D)

    E4 = round(m4 / (sigma ** 4) - 3, 2)
    return E4