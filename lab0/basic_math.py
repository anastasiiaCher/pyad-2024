import numpy as np
import scipy as sc

a = ''

def matrix_multiplication(matrix_a, matrix_b):
    # Проверка
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы нельзя перемножить")

    # Создание результирующей матрицы с нулями
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coefs1 = list(map(int, a_1.split()))
    coefs2 = list(map(int, a_2.split()))
    print(coefs1)
    extr1 = sc.optimize.minimize_scalar(lambda x: coefs1[0] * x **2 + coefs1[1] * x + coefs1[2])
    extr2 = sc.optimize.minimize_scalar(lambda x: coefs2[0] * x **2 + coefs2[1] * x + coefs2[2])
    print(extr1.x, extr2.x)

functions('1 0 -4', '1 -2 0')


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    pass
