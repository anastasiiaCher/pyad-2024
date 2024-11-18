import numpy as np
from scipy.optimize import fmin


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError('Число столбцов первой матрицы должно равняться числу строк второй матрицы')

    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] +=matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coef1 = list(map(float, a_1.split()))
    coef2 = list(map(float, a_2.split()))
    a11, a12, a13 = coef1
    a21, a22, a23 = coef2

    if coef1 == coef2:
        return None  # Если коэффициенты равны, бесконечно много решений

    def func1(x):
        return a11 * x ** 2 + a12 * x + a13

    def func2(x):
        return a21 * x ** 2 + a22 * x + a23

    extremum_F = fmin(func1, 0, disp=False)[0]
    extremum_P = fmin(func2, 0, disp=False)[0]

    # Параметры для нахождения пересечений
    A = a11 - a21
    B = a12 - a22
    C = a13 - a23

    if A == 0 and B == 0 and C == 0:
        return None  # Бесконечно много решений

    # Если функции имеют одинаковые коэффициенты, но разные свободные члены
    if A == 0 and B == 0 and C != 0:
        return []  # Нет решений, так как функции параллельны

    # Проверка на случай, когда это линейные уравнения
    if A == 0:
        if B != 0:
            return [(round(-C / B, 2), 0.0)]
        else:
            return None

    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        return []  # Нет решений
    elif discriminant == 0:
        # Одно решение (касание функций)
        x = -B / (2 * A)
        y = a11 * x ** 2 + a12 * x + a13
        return [(round(x, 2), round(y, 2))]
    else:
        # Два решения (пересечение функций)
        x1 = (-B + np.sqrt(discriminant)) / (2 * A)
        x2 = (-B - np.sqrt(discriminant)) / (2 * A)
        y1 = a11 * x1 ** 2 + a12 * x1 + a13
        y2 = a11 * x2 ** 2 + a12 * x2 + a13

        roots = [(round(x1, 2), round(y1, 2)), (round(x2, 2), round(y2, 2))]
        return roots


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    x_mean = np.mean(x) #выборочное среднее

    m2 = np.sum((x-x_mean) ** 2) / n #момент второго порядка
    m3 = np.sum((x-x_mean) ** 3) / n #момент третьего порядка

    sigma = np.sqrt(m2) #корень из дисперсии - стандартное отклонение
    skew1 = m3 / (sigma ** 3)

    return round(skew1, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    x_mean = np.mean(x)  # выборочное среднее

    m2 = np.sum((x - x_mean) ** 2) / n  # момент второго порядка
    m4 = np.sum((x - x_mean) ** 4) / n  # момент четвертого порядка

    sigma = np.sqrt(m2)  # корень из дисперсии - стандартное отклонение
    kurtosis1 = m4 / (sigma ** 4) - 3

    return round(kurtosis1, 2)
