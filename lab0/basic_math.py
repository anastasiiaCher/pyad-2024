import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar
from scipy.stats import skew as scipy_skew
from scipy.stats import kurtosis as scipy_kurtosis


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы.")

    # Размерность результата
    result_rows = len(matrix_a)
    result_cols = len(matrix_b[0])

    result = [[0 for _ in range(result_cols)] for _ in range(result_rows)] # 0 матрица

    for i in range(result_rows):
        for k in range(result_cols):
            for j in range(len(matrix_b)):
                result[i][k] += matrix_a[i][j] * matrix_b[j][k]
    
    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coeff1 = list(map(float, a_1.split()))
    coeff2 = list(map(float, a_2.split()))
    
    if coeff1 == coeff2:
        return None  
    

    def find_extremum(coeff):
        a, b, c = coeff
        if a == 0:
            return None  
        x_ext = -b / (2 * a)
        y_ext = a * x_ext**2 + b * x_ext + c
        return (round(x_ext), round(y_ext))  
    
    extremum1 = find_extremum(coeff1)
    extremum2 = find_extremum(coeff2)
    
    if extremum1 and extremum2 and extremum1 == extremum2:
        return [extremum1]  
    else:
        return []
coeffs1 = "1 0 -4"
coeffs2 = "1 -2 0"

print(functions(coeffs1, coeffs2))
def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    return round(scipy_skew(x), 2)



def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    return round(scipy_kurtosis(x), 2)

    
