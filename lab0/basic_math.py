import numpy as np
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
import random

def matrix_multiplication(A, B):
    # Проверяем, возможно ли умножение матриц
    if len(A[0]) != len(B):
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы")

    # Инициализируем результирующую матрицу нулями
    rows_A, cols_B, cols_A = len(A), len(B[0]), len(A[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Выполняем умножение матриц
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]


    return result

def parse_coefficients(input_str):
    return list(map(float, input_str.strip().split()))

def solve_quadratic(A, B, C):
    if A == 0:  # Линейное уравнение или константа
        if B == 0:
            return None if C == 0 else []  # Бесконечно много решений или их нет
        x = -C / B
        return [int(x)]

    # Решение квадратного уравнения
    D = B**2 - 4 * A * C  # Дискриминант
    if D > 0:
        x1 = (-B + np.sqrt(D)) / (2 * A)
        x2 = (-B - np.sqrt(D)) / (2 * A)
        return [int(x1), int(x2)]
    elif D == 0:
        x = -B / (2 * A)
        return [int(x)]
    else:
        return []

def calculate_difference_coefficients(coeffs1, coeffs2):
    return [a - b for a, b in zip(coeffs1, coeffs2)]

def evaluate_function(coeffs, x):
    a1, a2, a3 = coeffs
    return int(a1 * x**2 + a2 * x + a3)

def functions(coeffs1, coeffs2):
    # Парсинг коэффициентов
    parsed_coeffs1 = parse_coefficients(coeffs1)
    parsed_coeffs2 = parse_coefficients(coeffs2)

    # Вычисление разности коэффициентов
    A, B, C = calculate_difference_coefficients(parsed_coeffs1, parsed_coeffs2)

    # Решение уравнения F(x) - P(x) = 0
    roots = solve_quadratic(A, B, C)

    if roots is None:  # Бесконечно много решений
        return None
    if not roots:  # Нет решений
        return []

    # Вычисление значений F(x) или P(x) для каждого корня
    results = [(x, evaluate_function(parsed_coeffs1, x)) for x in roots]
    return sorted(results)


def calculate_moments(sample):
    # Вычисляем выборочное среднее
    x_mean = np.mean(sample)

    # Общий объем выборки
    n = len(sample)

    # Центральные моменты
    m2 = np.sum((sample - x_mean)**2) / n
    m3 = np.sum((sample - x_mean)**3) / n
    m4 = np.sum((sample - x_mean)**4) / n

    return x_mean, m2, m3, m4

def skew(sample):
    # Рассчитаем моменты
    x_mean, m2, m3, m4 = calculate_moments(sample)

    # Стандартное отклонение
    sigma = np.sqrt(m2)

    # Коэффициент асимметрии
    A3 = m3 / sigma**3

    # Округляем результат до 2 знаков после запятой и возвращаем
    return round(A3, 2)

def kurtosis(sample):
    # Рассчитаем моменты
    x_mean, m2, m3, m4 = calculate_moments(sample)

    # Стандартное отклонение
    sigma = np.sqrt(m2)

    # Коэффициент эксцесса
    E4 = m4 / sigma**4 - 3

    # Округляем результат до 2 знаков после запятой и возвращаем
    return round(E4, 2)
