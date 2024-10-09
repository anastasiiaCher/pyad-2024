import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы")

    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    multiplication = [[0 for i in range(cols_b)] for j in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                multiplication[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return multiplication


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    delta_a2 = a11 - a21
    delta_a1 = a12 - a22
    delta_a0 = a13 - a23

    if delta_a2 == 0 and delta_a1 == 0 and delta_a0 == 0:
        return None

    if delta_a2 == 0:
        if delta_a1 == 0:
            return []  
        else:
            root = -delta_a0 / delta_a1
            return [(round(root), round(a11 * root**2 + a12 * root + a13))] 

    discriminant = delta_a1**2 - 4 * delta_a2 * delta_a0

    if discriminant < 0:
        return [] 

    elif discriminant == 0:
        root = -delta_a1 / (2 * delta_a2)
        return [(round(root), round(a11 * root**2 + a12 * root + a13))]

    else:
        root1 = (-delta_a1 + np.sqrt(discriminant)) / (2 * delta_a2)
        root2 = (-delta_a1 - np.sqrt(discriminant)) / (2 * delta_a2)
        return [(round(root1), round(a11 * root1**2 + a12 * root1 + a13)),
                (round(root2), round(a11 * root2**2 + a12 * root2 + a13))]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean_x = sum(x) / n 

    m2 = sum((i - mean_x) ** 2 for i in x) / n  
    m3 = sum((i - mean_x) ** 3 for i in x) / n 

    skew_value = m3 / (m2 ** 1.5)
    return round(skew_value, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    x_mean = sum(x) / n 

    m2 = sum((i - x_mean) ** 2 for i in x) / n 
    m4 = sum((i - x_mean) ** 4 for i in x) / n 

    kurtosis_value = m4 / (m2 ** 2) - 3
    return round(kurtosis_value, 2)
