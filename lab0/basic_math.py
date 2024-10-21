import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # Проверяем, можно ли умножать матрицы (количество столбцов A должно быть равно количеству строк B)
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError(
            "Матрицы нельзя умножить: количество столбцов первой матрицы должно быть равно количеству строк второй."
        )

    # Получаем размеры матриц
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    cols_b = len(matrix_b[0])

    # Создаем результирующую матрицу с размерами rows_A x cols_B и заполняем нулями
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    # Выполняем умножение матриц
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):  # или len(B), что то же самое
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def f(x, a11, a12, a13):
    return a11 * x ** 2 + a12 * x + a13


def p(x, a21, a22, a23):
    return a21 * x ** 2 + a22 * x + a23


def find_extremum(coeffs, func):
    result = sc.optimize.minimize_scalar(func, args=tuple(coeffs))
    return result.x, result.fun


def find_intersection(coeffs_f, coeffs_p):
    # Приведем уравнение F(x) - P(x) = 0 к виду a*x^2 + b*x + c = 0
    a = coeffs_f[0] - coeffs_p[0]
    b = coeffs_f[1] - coeffs_p[1]
    c = coeffs_f[2] - coeffs_p[2]

    if a == 0:
        if b == 0:
            if c == 0:
                return None  # F(x) = P(x) для всех x
            else:
                return [] # Нет решений, так как уравнение сводится к c = 0, а c ≠ 0
        else:
            # Линейное уравнение b*x + c = 0
            x = -c / b
            return [x]
    else:
        # Квадратичное уравнение: a*x^2 + b*x + c = 0
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


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coeffs_f = list(map(float, a_1.split()))
    coeffs_p = list(map(float, a_2.split()))

    # extremum_f_x, extremum_f_value = find_extremum(coeffs_f, f)
    # extremum_p_x, extremum_p_value = find_extremum(coeffs_p, p)

    intersections = find_intersection(coeffs_f, coeffs_p)

    if intersections is None:
        return None
    else:
        return list(map(lambda x: tuple((x, f(x, *coeffs_f))) if x is not None else None, intersections))


def calculate_moments(sample):
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample, ddof=0)
    m3 = np.sum((sample - mean) ** 3) / n
    m4 = np.sum((sample - mean) ** 4) / n
    return m3, m4, std


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m3, m4, std = calculate_moments(x)
    return round(m3 / (std ** 3), 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m3, m4, std = calculate_moments(x)
    return round(m4 / (std ** 4) - 3, 2)

