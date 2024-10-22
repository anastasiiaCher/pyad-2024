import numpy as np
import scipy as sc



def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # put your code here
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError
    n, m, k = len(matrix_a), len(matrix_b), len(matrix_b[0])
    matrix = np.zeros((n, k), dtype=int)

    for i in range(n):
        for j in range(k):
            for p in range(m):
                matrix[i][j] += matrix_a[i][p]*matrix_b[p][j]

    return matrix.tolist()


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # put your code here
    a_1 = [int(i) for i in a_1.split()]
    a_2 = [int(i) for i in a_2.split()]
    if a_2 == a_1:
        return None
    D = (a_1[1]-a_2[1])**2 - 4*(a_1[0]-a_2[0])*(a_1[2]-a_2[2])
    if (a_1[0]-a_2[0]) == 0:
        if (a_1[1]-a_2[1]) == 0: return []
        x1 = (-a_1[2]+a_2[2]) / (a_1[1]-a_2[1])
        return [(round(x1, 3), round(a_1[0]*x1*x1+a_1[1]*x1+a_1[2], 3))]
    if D > 0:
        x1 = ((-a_1[1]+a_2[1]) + D**0.5)/(2*((a_1[0]-a_2[0])))
        x2 = ((-a_1[1]+a_2[1]) - D**0.5)/(2*((a_1[0]-a_2[0])))
        return [(round(x1, 3), round(a_1[0]*x1*x1+a_1[1]*x1+a_1[2], 3)), (round(x2, 3), round(a_1[0]*x2*x2+a_1[1]*x2+a_1[2], 3))]
    elif D == 0:
        x1 = ((-a_1[1]+a_2[1]) + D**0.5)/2*((a_1[0]-a_2[0]))
        x2 = x1
        return [(round(x1, 3), round(a_1[0]*x1*x1+a_1[1]*x1+a_1[2], 3))]
    else:
        return []


def fun(X):
    N = len(X)
    x_e = 0
    for i in set(X):
        x_e += (i * X.count(i))/N

    m2, m3, m4 = 0, 0, 0

    for i in set(X):
        m2 += ((i - x_e)**2) * X.count(i)/N
        m3 += ((i - x_e)**3) * X.count(i)/N
        m4 += ((i - x_e)**4) * X.count(i)/N

    A = m3/(m2**1.5)
    E = (m4/(m2**2)) - 3
    return (A, E)

def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    return round(fun(x)[0], 2)

def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    return round(fun(x)[1], 2)
