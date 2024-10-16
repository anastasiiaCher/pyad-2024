import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    matrixA_size = [len(matrix_a), len(matrix_a[0])];
    matrixB_size = [len(matrix_b), len(matrix_b[0])]

    if (matrixA_size[1] != matrixB_size[0]):
        raise ValueError('Incorrect matrix size')

    result_matrix = []

    for i in range(len(matrix_a)):
        result_matrix.append([])

        for k in range(len(matrix_b[0])):
            accumulator = 0

            for j in range(len(matrix_a[0])):
                accumulator += matrix_a[i][j] * matrix_b[j][k]

            result_matrix[i].append(accumulator)
    
    return result_matrix


def functions(fn1, fn2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    if fn1 == fn2:
        return None
  
    a11, a12, a13 = map(float, fn1.split())
    a21, a22, a23 = map(float, fn2.split())


    a = a11 - a21
    b = a12 - a22
    c = a13 - a23

    def F(x):
        return a11 * x ** 2 + a12 * x + a13

    discr = b ** 2 - 4 * a * c

    isLinear = a == 0

    if isLinear:
        hasRoot = b != 0
        
        if not hasRoot:
            return []

        x = - c / b
        return [(x, F(x))]
    
    else:
        if discr < 0:
            return []
        
        if discr == 0:
            x = (-b + discr ** 0.5 ) / (2 * a)
            return [x, F(x)]
        
        x1 = (-b + discr ** 0.5) / (2 * a)
        x2 = (-b - discr ** 0.5) / (2 * a)

        return [(x1, F(x1)), (x2, F(x2))]


def skew(data):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    mean = sum(data) / len(data)
    third_moment = sum([(elem - mean) ** 3 for elem in data]) / len(data)
    sigma = (sum([(elem - mean) ** 2 for elem in data]) / len(data)) ** (1 / 2)

    return round(third_moment / sigma ** 3, 2)


def kurtosis(data):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    mean = sum(data) / len(data)
    fourth_moment = sum([(elem - mean) ** 4 for elem in data]) / len(data)
    sigma = (sum([(elem - mean) ** 2 for elem in data]) / len(data)) ** (1 / 2)

    return round(fourth_moment / sigma ** 4 - 3, 2)
