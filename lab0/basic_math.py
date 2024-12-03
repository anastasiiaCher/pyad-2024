import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar, fsolve
import random


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError()
    result_multiplication= [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):  
        for j in range(len(matrix_b[0])):  
            for k in range(len(matrix_b)): 
                result_multiplication[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result_multiplication


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    # Определим функции
    def F(x):
        return a11 * x**2 + a12 * x + a13

    def P(x):
        return a21 * x**2 + a22 * x + a23

    # Поиск экстремума через -b / (2a):
    def extremum(a, b):
        return -b / (2 * a)

    if a11 != 0:
        extremum_F = extremum(a11, a12)
    else:
        None
    if a21 != 0:
        extremum_P = extremum(a21, a22)
    else:
        None


    # Решаем уравнение F(x) = P(x) и находим дискриминант
    A = a11 - a21
    B = a12 - a22
    C = a13 - a23

    discriminant = B**2 - 4 * A * C

    # Анализируем дискриминант для определения количества решений
    if A == 0 and B == 0 and C == 0:
        return None
    elif A == 0 and B == 0:
        return []
    elif A == 0:
        x = -C / B
        return [(x, F(x))]
    elif discriminant > 0:
        x1 = (-B + np.sqrt(discriminant)) / (2 * A)
        x2 = (-B - np.sqrt(discriminant)) / (2 * A)
        return [(x1, F(x1)), (x2, F(x2))]
    elif discriminant == 0:
        x = -B / (2 * A)
        return [(x, F(x))]
    else:
        return None


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    sample_mean = sum(x)/len(x)
    m_3 = sum(((i - sample_mean)** 3 for i in x)) / len(x)  
    s = np.sqrt((1/len(x)) *sum((i - sample_mean)**2 for i in x ))
    coefficient_of_asymmetry = round(m_3/(s**3),2)
    return coefficient_of_asymmetry


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    sample_mean = sum(x)/len(x)
    m_4 = sum(((i - sample_mean) ** 4 for i in x)) / len(x)  
    s = np.sqrt( (1/len(x)) *sum((i - sample_mean)**2 for i in x ))
    kurtosis_coefficient = round(((m_4/(s**4))-3),2)
    return kurtosis_coefficient
