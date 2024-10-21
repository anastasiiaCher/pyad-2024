import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов 1-ой матрицы должно быть равно числу строк 2-ой матрицы")

    num_rows_a = len(matrix_a)  # Количество строк в первой матрице
    num_cols_b = len(matrix_b[0])  # Количество столбцов во второй матрице
    num_common = len(matrix_b)  # Количество столбцов первой матрицы = числу строк второй матрицы

    result = [[0] * num_cols_b for i in range(num_rows_a)]

    for i in range(num_rows_a):
        for j in range(num_cols_b):
            result[i][j] = sum(matrix_a[i][k] * matrix_b[k][j] for k in range(num_common))

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a1 = list(map(float, a_1.split()))
    a2 = list(map(float, a_2.split()))

    a = a1[0] - a2[0]  # Разница коэффициентов при x^2
    b = a1[1] - a2[1]  # Разница коэффициентов при x
    c = a1[2] - a2[2]  # Разница свободных коэффициентов

    if a == 0 and b == 0 and c == 0:
        return None

    if a == 0:
        if b == 0:
            return []
        else:
            x = -c / b
            return [(x, a1[0] * x ** 2 + a1[1] * x + a1[2])]

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return []

    elif discriminant == 0:
        x = -b / (2 * a)
        return [(x, a1[0] * x ** 2 + a1[1] * x + a1[2])]

    else:
        x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        x2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return [(x1, a1[0] * x1 ** 2 + a1[1] * x1 + a1[2]), (x2, a1[0] * x2 ** 2 + a1[1] * x2 + a1[2])]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    mean = np.mean(x)
    m2 = np.mean((x - mean) ** 2)  # Дисперсия
    m3 = np.mean((x - mean) ** 3)  # Третий момент

    sigma = np.sqrt(m2)  # Стандартное отклонение
    skewness = m3 / sigma ** 3  # Коэффициент асимметрии

    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    mean = np.mean(x)
    m2 = np.mean((x - mean) ** 2)  # Дисперсия
    m4 = np.mean((x - mean) ** 4)  # Четвертый момент

    sigma = np.sqrt(m2)  # Стандартное отклонение
    excess_kurtosis = (m4 / sigma ** 4) - 3  # Коэффициент эксцесса

    return round(excess_kurtosis, 2)
