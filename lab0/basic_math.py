import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Matrices are incompatible for multiplication")
    len1, len2, len3 = len(matrix_a), len(matrix_b), len(matrix_b[0])
    matrix = [[0] * len3 for _ in range(len1)]
    for i in range(len1):
        for k in range(len3):
            matrix[i][k] = sum(matrix_a[i][j] * matrix_b[j][k] for j in range(len2))

    return matrix


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coeff_F = a_1.split()
    coeff_P = a_2.split()
    a11, a12, a13 = map(float, coeff_F)
    a21, a22, a23 = map(float, coeff_P)

    a = a11 - a21
    b = a12 - a22
    c = a13 - a23
    # необходимо проверить, не 0 ли а, тк иначе получим ошибку при делении на 0.
    if a == 0:
        if b == 0:
            if c == 0:
                return None
            else:
                return []
        else:
            x = -c / b
            return [x]
    else:
        D = b ** 2 - 4 * a * c
        if D > 0:
            x1 = (-b + np.sqrt(D)) / (2 * a)
            x2 = (-b - np.sqrt(D)) / (2 * a)
            return [x1, x2]
        elif D == 0:
            x = -b / (2 * a)
            return [x]
        else:
            return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    x = np.array(x)
    n = len(x)
    mean = np.mean(x)
    m3 = np.mean((x - mean) ** 3)
    m2 = np.mean((x - mean) ** 2)
    a3 = m3 / (m2 ** 1.5) if m2 > 0 else 0
    return round(a3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x = np.array(x)
    n = len(x)
    mean = np.mean(x)
    m4 = np.mean((x - mean) ** 4)
    m2 = np.mean((x - mean) ** 2)
    e4 = m4 / (m2 ** 2) - 3 if m2 > 0 else 0
    return round(e4, 2)
