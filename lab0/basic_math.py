import math
from typing import List
import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError()

    # Транспонирование матрицы
    matrix_b_T = [[matrix_b[j][i] for j in range(len(matrix_b))] for i in range(len(matrix_b[0]))]

    matrix_result: List[List[float]] = []

    for row_a in matrix_a:
        row = []
        for column_b in matrix_b_T:
            row.append(sum(row_a[i] * column_b[i] for i in range(len(row_a))))
        matrix_result.append(row)
    return matrix_result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    def find_extremum(a, b, c):
        def f(x):
            return a * x ** 2 + b * x + c

        res = minimize_scalar(f, bounds=(-1000, 1000), method='bounded')
        return res.x, f(res.x)

    a11, a12, a13 = map(float, a_1.strip().split())
    a21, a22, a23 = map(float, a_2.strip().split())

    def F(x):
        return a11 * x ** 2 + a12 * x + a13

    def P(x):
        return a21 * x ** 2 + a22 * x + a23

    a_1_extremum = find_extremum(a11, a12, a13)
    a_2_extremum = find_extremum(a21, a22, a23)

    print(f"Экстремум первой функции: {a_1_extremum}")
    print(f"Экстремум второй функции: {a_2_extremum}")

    A = a11 - a21
    B = a12 - a22
    C = a13 - a23

    # Решаем получившееся квадратное уравнение
    discriminant = B ** 2 - 4 * A * C
    if A == 0 and B == 0 and C == 0:
        print("Бесконечно много решений")
        return None
    elif A == 0:
        # Линейное уравнение
        if B != 0:
            print(f"Одно решение {-C / B}")
            return [(-C / B, F(-C / B))]
        else:
            print(f"Нет решений")
            return []
    else:
        # Квадратное уравнение
        if discriminant < 0:
            print(f"Нет реальных корней")
            return []
        elif discriminant == 0:
            print(f"Одно решение {-B / (2 * A)}")
            return [(-B / (2 * A), F(-B / (2 * A)))]
        else:
            root1 = (-B + np.sqrt(discriminant)) / (2 * A)
            root2 = (-B - np.sqrt(discriminant)) / (2 * A)
            print(f"Два решения {root1} и {root2}")
            return [(root1, F(root1)), (root2, F(root2))]

def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean_x = np.mean(x) # Первый момент
    m2 = np.sum((x - mean_x) ** 2) / n  # Второй момент
    m3 = np.sum((x - mean_x) ** 3) / n  # Третий момент
    skewness = m3 / (m2 ** (3 / 2))
    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x_mean = np.mean(x)  # Выборочное среднее
    m2 = np.mean((x - x_mean) ** 2)  # Момент второго порядка
    m4 = np.mean((x - x_mean) ** 4)  # Момент четвертого порядка
    kurt = m4 / (m2 ** 2) - 3
    return round(kurt, 2)