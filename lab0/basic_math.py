import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
            raise ValueError()

    matrix_result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_a[0])):
            for k in range(len(matrix_b[0])):
                matrix_result[i][k] += matrix_a[i][j] * matrix_b[j][k]
    return matrix_result



def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    index1 = [int(a) for a in a_1.split()]
    index2 = [int(a) for a in a_2.split()]

    if index1 == index2:
        return None

    def F(x, index=index1):
        return index[0] * x ** 2 + index[1] * x + index[2]

    def P(x, index=index2):
        return index[0] * x ** 2 + index[1] * x + index[2]

    a = index1[0] - index2[0]
    b = index1[1] - index2[1]
    c = index1[2] - index2[2]
    if a == 0 and b != 0:
        root = -c/b
        return [(root, F(root))]
    elif a == 0:
        return []
    else:
        D = b ** 2 - 4 * a * c

        if D < 0:
            return []

        root1 = (-b + (D**0.5))/(2*a)
        root2 = (-b - (D**0.5))/(2*a)
        if D == 0:
            return [(root1, F(root1))]
        return [(root1, F(root1)), (root2, F(root2))]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    x_e = sum(x) / len(x)
    sigma = (sum([(num - x_e) ** 2 for num in x]) / len(x)) ** 0.5
    m_3 = sum([(num - x_e) ** 3 for num in x]) / len(x)
    A_3 = m_3 / (sigma ** 3)
    return round(A_3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x_e = sum(x) / len(x)
    sigma = (sum([(num - x_e) ** 2 for num in x]) / len(x)) ** 0.5
    m_4 = sum([(num - x_e) ** 4 for num in x]) / len(x)
    E_4 = m_4 / (sigma ** 4) - 3
    return round(E_4, 2)
