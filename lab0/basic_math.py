import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar, fsolve
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # Проверка возможности умножения
    if len(matrix_a[0]) != len(matrix_b):
        return "Умножение невозможно: число столбцов первой матрицы не равно числу строк второй матрицы."
    
    # Инициализация результирующей матрицы нулями
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    # Умножение матриц через тройной цикл
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
    # Преобразование строк в коэффициенты
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())
    
    # Определение функций F(x) и P(x)
    def F(x):
        return a11 * x**2 + a12 * x + a13

    def P(x):
        return a21 * x**2 + a22 * x + a23

    # Нахождение экстремумов через минимизацию
    res_F = minimize_scalar(F)
    res_P = minimize_scalar(P)
    
    # Поиск общих решений: F(x) = P(x)
    def equation(x):
        return F(x) - P(x)
    
    # Поиск решения
    solution = fsolve(equation, x0=0)
    
    if np.isclose(F(solution), P(solution)):
        return [solution[0]]
    else:
        return None


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # Вычисляем коэффициент асимметрии с помощью scipy
    skewness = scipy_skew(x)
    
    # Округляем до 2 знаков после запятой
    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # Вычисляем коэффициент эксцесса с помощью scipy
    excess = scipy_kurtosis(x)
    
    # Округляем до 2 знаков после запятой
    return round(excess, 2)
