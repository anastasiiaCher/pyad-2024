import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы.")

    rows_A = len(matrix_a)  # Число строк первой матрицы
    cols_A = len(matrix_a[0])  # Число столбцов первой матрицы (и строк второй матрицы)
    rows_B = len(matrix_b)  # Число строк второй матрицы (это то же самое, что cols_A)
    cols_B = len(matrix_b[0])  # Число столбцов второй матрицы

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    if a_1 == a_2:
        return None
    
    a1, b1, c1 = map(float, a_1.split())
    a2, b2, c2 = map(float, a_2.split())

    # Поиск экстремумов
    if a1 != 0:
        minimum_1 = -b1 / (2 * a1)
    if a2 != 0:
        minimum_2 = -b2 / (2 * a2)
    
    # Разность функций: (a1 - a2)x^2 + (b1 - b2)x + (c1 - c2) = 0
    a = a1 - a2
    b = b1 - b2
    c = c1 - c2
    
    # Если уравнение имеет степень 2
    if a != 0:
        discriminant = b**2 - 4 * a * c
        
        # Если дискриминант равен 0 — одно решение
        if discriminant == 0:
            x = -b / (2 * a)
            return [(x, a1 * x**2 + b1 * x + c1)]
        # Если дискриминант отрицателен — решений нет
        elif discriminant < 0:
            return []
        # Если дискриминант положителен — два решения
        else:
            sqrt_d = discriminant**0.5
            x1 = (-b + sqrt_d) / (2 * a)
            x2 = (-b - sqrt_d) / (2 * a)
            return [(x1, a1 * x1**2 + b1 * x1 + c1), (x2, a1 * x2**2 + b1 * x2 + c1)]
    
    elif b != 0:
        x = -c / b
        return [(x, a1 * x**2 + b1 * x + c1)]

    else:
        return []
    
def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    
    mean_x = sum(x) / n 

    m2 = sum((i - mean_x) ** 2 for i in x) / n  
    m3 = sum((i - mean_x) ** 3 for i in x) / n 

    value = m3 / (m2 ** 1.5)
    return round(value, 2)

def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)

    x_mean = sum(x) / n 

    m2 = sum((i - x_mean) ** 2 for i in x) / n 
    m4 = sum((i - x_mean) ** 4 for i in x) / n 

    value = m4 / (m2 ** 2) - 3
    return round(value, 2)
