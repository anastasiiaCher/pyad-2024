import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    result = []

    for _ in range(len(matrix_a)):
        result.append([0] * len(matrix_b[0]))

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов матрицы_1 не совпадает с числом строк матрицы_2")

    return result

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    solutions = []
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    a = a11-a21
    b = a12-a22
    c = a13-a23
    discriminant = b**2-4*a*c

    if a == 0:
        if b != 0:
            x = -c/b
            solutions.append((x, a11*x**2+a12*x+a13))
    elif discriminant > 0:
        x1 = (-b + (discriminant)**0.5)/(2*a)
        x2 = (-b - (discriminant)**0.5)/(2*a)
        solutions.append((x1, a11*x1**2+a12*x1+a13))
        solutions.append((x2, a11*x2**2+a12*x2+a13))
    elif discriminant == 0:
        x = -b/(2*a)
        solutions.append((x, a11*x**2+a12*x+a13))
    else:
        solutions = None

    if a_1 == a_2:
        return None

    return solutions



def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    k = len(x)
    mean = np.mean(x)
    std = np.std(x)
    skewness = (sum((x - mean)**3)/k)/(std**3)

    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    k = len(x)
    mean = np.mean(x)
    std = np.std(x)
    kurt = (sum((x-mean)**4)/k)/(std**4)-3

    return round(kurt, 2)
