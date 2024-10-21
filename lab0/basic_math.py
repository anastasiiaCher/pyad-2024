import numpy as np


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # Проверка совместимости матриц
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError('Error')

    rows_a = len(matrix_a)
    cols_b = len(matrix_b[0])

    multiply = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(len(matrix_b)):
                multiply[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return multiply


def functions(a_1, a_2):
    """
    На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    def parse_coefficients(coeffs_str):
        return list(map(float, coeffs_str.split()))

    def quadratic_function(a, b, c):
        return lambda x: a * x ** 2 + b * x + c

    def find_extremum(func):
        a = func(1) - func(0)
        b = func(0) - func(1)
        extremum_x = -b / (2 * a)
        extremum_y = func(extremum_x)
        return round(extremum_x, 2), round(extremum_y, 2)

    def find_intersections(a1, b1, c1, a2, b2, c2):
        """Находит точки пересечения двух квадратичных функций."""
        A = a1 - a2
        B = b1 - b2
        C = c1 - c2

        if A == 0 and B == 0 and C == 0:  # Функции идентичны
            return None

        if A == 0:  # Линейные функции
            if B == 0:  # Параллельные функции
                return []
            root = -C / B
            return [(round(root, 2), round(quadratic_function(a1, b1, c1)(root), 2))]

        discriminant = B ** 2 - 4 * A * C

        if discriminant < 0:
            return []  # Нет пересечений
        elif discriminant == 0:
            root = -B / (2 * A)
            return [(round(root, 2), round(quadratic_function(a1, b1, c1)(root), 2))]

        root1 = (-B + np.sqrt(discriminant)) / (2 * A)
        root2 = (-B - np.sqrt(discriminant)) / (2 * A)

        return [
            (round(root1, 2), round(quadratic_function(a1, b1, c1)(root1), 2)),
            (round(root2, 2), round(quadratic_function(a1, b1, c1)(root2), 2))
        ]

    a1, b1, c1 = parse_coefficients(a_1)
    a2, b2, c2 = parse_coefficients(a_2)

    F = quadratic_function(a1, b1, c1)
    P = quadratic_function(a2, b2, c2)

    extremum_f = find_extremum(F)
    extremum_p = find_extremum(P)

    intersections = find_intersections(a1, b1, c1, a2, b2, c2)

    return intersections


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = sum(x) / n

    variance = sum((i - mean) ** 2 for i in x) / n
    std = variance ** 0.5

    # Коэффициент эксцесса
    skewness = sum((i - mean) ** 3 for i in x) / (n * std ** 3)

    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = sum(x) / n

    variance = sum((i - mean) ** 2 for i in x) / n
    std = variance ** 0.5

    # Коэффициент эксцесса
    kurtosis = sum((i - mean) ** 4 for i in x) / (n * std ** 4) - 3

    return round(kurtosis, 2)
