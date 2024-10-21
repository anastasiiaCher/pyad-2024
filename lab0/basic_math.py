import numpy as np
import scipy as sc
"""
### Задание 1
Функция для перемножения матриц с помощью списков и циклов. Вернуть нужно матрицу в формате списка.
Для начала проверим, совпадает ли число столбцов первой матрицы с числом строк второй.
Далее – создадим результирующую матрицу в виде списка, в который будем заносить ответы. И перемножим заданные матрицы.
"""
def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError()
    
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

"""
### Задание 2
На вход поступает две строки, содержащие коэффициенты двух функций. Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения. Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
Итого нам нужно:
- Найти точки экстремума функций. То есть необходимо найти производную функций и решение уравнений, где производная равна 0.
- Найти общие решения этих двух функций. То есть нам нужно решить систему уравнений f(x) = g(x), что, соответственно, сводится к решению квадратного уравнения.
"""
def functions(a_1, a_2):
    coeffs1 = list(map(int, a_1.split()))
    coeffs2 = list(map(int, a_2.split()))
    
    a1, b1, c1 = coeffs1
    a2, b2, c2 = coeffs2
    
    if coeffs1 == coeffs2:
        return None
    
    A = a1 - a2
    B = b1 - b2
    C = c1 - c2
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        return []
    elif discriminant == 0:
        x = -B / (2*A)
        y = a1*x**2 + b1*x + c1
        return [(x, y)]
    else:
        x1 = (-B + np.sqrt(discriminant)) / (2*A)
        x2 = (-B - np.sqrt(discriminant)) / (2*A)
        y1 = a1*x1**2 + b1*x1 + c1
        y2 = a1*x2**2 + b1*x2 + c1
        return [(x1, y1), (x2, y2)]
        
"""
### Задание 3
3.1. Функция для расчета коэффициента асимметрии. Необходимо вернуть значение коэффициента асимметрии,
округленное до 2 знаков после запятой.
"""
def skew(x):
    n = len(x)
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    skewness = (n / ((n - 1) * (n - 2))) * sum(((xi - mean_x) / std_x) ** 3 for xi in x)
    return round(skewness, 2)

"""
3.2. Функция для расчета коэффициента эксцесса. Необходимо вернуть значение коэффициента
эксцесса, округленное до 2 знаков после запятой.
"""
def kurtosis(x):
    n = len(x)
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    kurt = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum(((xi - mean_x) / std_x) ** 4 for xi in x)
    kurt -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return round(kurt, 2)
