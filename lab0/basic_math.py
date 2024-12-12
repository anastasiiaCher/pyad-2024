import numpy as np
import scipy as sc

def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # Проверка совместимости матриц
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы нельзя перемножить: число столбцов первой матрицы не равно числу строк второй матрицы.")

    # Результирующая матрица
    result = [[sum(matrix_a[i][k] * matrix_b[k][j] for k in range(len(matrix_b))) for j in range(len(matrix_b[0]))] for i in range(len(matrix_a))]
    return result

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    from numpy import isclose

    # Разбираем коэффициенты
    coeff1 = list(map(float, a_1.split()))
    coeff2 = list(map(float, a_2.split()))

    # Определяем функции
    func1 = lambda x: coeff1[0] * x**2 + coeff1[1] * x + coeff1[2]
    func2 = lambda x: coeff2[0] * x**2 + coeff2[1] * x + coeff2[2]

    # Проверка на идентичность функций
    if coeff1 == coeff2:
        return None

    # Решение уравнения func1(x) = func2(x)
    # Приводим уравнение к виду (a1 - a2)x^2 + (b1 - b2)x + (c1 - c2) = 0
    a = coeff1[0] - coeff2[0]
    b = coeff1[1] - coeff2[1]
    c = coeff1[2] - coeff2[2]

    # Если a, b, c равны 0, функций совпадают
    if a == 0 and b == 0 and c == 0:
        return None

    # Если a и b равны 0, решений нет
    if a == 0 and b == 0:
        return []

    # Если a == 0, решаем линейное уравнение bx + c = 0
    if a == 0:
        x = -c / b
        return [(x, func1(x))]

    # Решаем квадратное уравнение
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []  # Нет действительных корней
    elif isclose(discriminant, 0):
        x = -b / (2 * a)
        return [(x, func1(x))]
    else:
        x1 = (-b + discriminant**0.5) / (2 * a)
        x2 = (-b - discriminant**0.5) / (2 * a)
        return [(x1, func1(x1)), (x2, func1(x2))]

def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    x = np.array(x)
    mean = np.mean(x)
    m3 = np.mean((x - mean)**3)
    variance = np.var(x, ddof=0)
    skewness = m3 / variance**(3/2)
    return round(skewness, 2)

def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x = np.array(x)
    mean = np.mean(x)
    m4 = np.mean((x - mean)**4)
    variance = np.var(x, ddof=0)
    kurtosis_value = m4 / variance**2 - 3
    return round(kurtosis_value, 2)