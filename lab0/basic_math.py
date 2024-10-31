import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError()
    else:
        result = [[0 for i in range(len(matrix_b[0]))] for i in range(len(matrix_a))]

        for i in range(len(matrix_a)):  
            for j in range(len(matrix_b[0])):
                for k in range(len(matrix_b)):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(coeffs_F, coeffs_P):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # Записывает коэффициенты из строки
    a11, a12, a13 = map(float, coeffs_F.split())
    a21, a22, a23 = map(float, coeffs_P.split())

    # Определим функции
    def F(x):
        return a11 * x**2 + a12 * x + a13

    def P(x):
        return a21 * x**2 + a22 * x + a23

    # Поиск экстремума через -b / (2a):
    def extremum(a, b):
        return -b / (2 * a)

    # Находим экстремумы
    if a11 != 0:
        extremum_F = extremum(a11, a12)
    else:
        None
    if a21 != 0:
        extremum_P = extremum(a21, a22)
    else:
        None


    # Решаем уравнение F(x) = P(x) и находим дискриминант
    A = a11 - a21
    B = a12 - a22
    C = a13 - a23

    discriminant = B**2 - 4 * A * C

    # Анализируем дискриминант для определения количества решений
    if A == 0 and B == 0 and C == 0:
        return None
    elif A == 0 and B == 0:
        return []
    elif A == 0:
        x = -C / B
        return [(x, F(x))]
    elif discriminant > 0:
        x1 = (-B + np.sqrt(discriminant)) / (2 * A)
        x2 = (-B - np.sqrt(discriminant)) / (2 * A)
        return [(x1, F(x1)), (x2, F(x2))]
    elif discriminant == 0:
        x = -B / (2 * A)
        return [(x, F(x))]
    else:
        return None


def skew(sample):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(sample)  # Объем выборки
    mean = np.mean(sample)  # Выборочное среднее

    # Центральные моменты
    m2 = np.sum((sample - mean)**2) / n  # Момент второго порядка (дисперсия)
    m3 = np.sum((sample - mean)**3) / n  # Момент третьего порядка

    # Стандартное отклонение (корень из m2)
    sigma = np.sqrt(m2)

    # Коэффициент асимметрии A3
    A3 = m3 / sigma**3

    # Вывод результатов
    return round(A3, 2)


def kurtosis(sample):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(sample)  # Объем выборки
    mean = np.mean(sample)  # Выборочное среднее

    # Центральные моменты
    m2 = np.sum((sample - mean)**2) / n  # Момент второго порядка (дисперсия
    m4 = np.sum((sample - mean)**4) / n  # Момент четвертого порядка

    # Стандартное отклонение (корень из m2)
    sigma = np.sqrt(m2)

    # Коэффициент эксцесса E4
    E4 = m4 / sigma**4 - 3

    # Вывод результатов
    return round(E4, 2)
