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
    a1 = list(map(float, a_1.split()))
    a2 = list(map(float, a_2.split()))

    a_11, a_12, a_13 = a1
    a_21, a_22, a_23 = a2
    

    if a_11 != 0: 
        vertex_F = -a_12 / (2 * a_11)
    if a_21 != 0:
        vertex_P = -a_22 / (2 * a_21)
    coeffs_diff = [a_11 - a_21, a_12 - a_22, a_13 - a_23]
    roots = np.roots(coeffs_diff)
    solutions = []
    for root in roots:
        if np.isreal(root): 
            x = root.real
            
            y = a_11 * x**2 + a_12 * x + a_13
            solutions.append((round(x), round(y)))

    
    if all(coef == 0 for coef in coeffs_diff):
        return None  

    return solutions if solutions else None


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
