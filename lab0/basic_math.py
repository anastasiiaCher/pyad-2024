import numpy as np


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы.")

    n = len(matrix_a)
    m = len(matrix_b[0])
    result = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coeffs1 = [int(coeff) for coeff in a_1.split()]
    coeffs2 = [int(coeff) for coeff in a_2.split()]

    # находим точки экстремума:
    extreme_point_1 = get_extreme(coeffs1)
    extreme_point_2 = get_extreme(coeffs2)
    print(extreme_point_1)
    print(extreme_point_2)

    common_x = find_common_x(coeffs1, coeffs2)
    if common_x is None:
        return None
    return [(x, get_y(x, coeffs1)) for x in common_x]


def get_extreme(coeffs):
    a, b, c = coeffs

    if a == 0:
        return None
    x_extreme = -b / (2 * a)
    y_extreme = a * x_extreme * x_extreme + b * x_extreme + c
    return x_extreme, y_extreme


def find_common_x(coeffs1, coeffs2):
    # находим общие решения:
    a = coeffs1[0] - coeffs2[0]
    b = coeffs1[1] - coeffs2[1]
    c = coeffs1[2] - coeffs2[2]

    if a == b == 0:
        # проверка на бесконечное количество решений:
        if c == 0:
            return None
        # корни отсутствуют:
        return []

    # уравнение линейное -> одно решение
    if a == 0:
        x = -c / b
        return [x]

    # квадратное уравнение -> 0, 1 или 2 корня
    D = b * b - 4 * a * c
    if D < 0:
        return []
    elif D == 0:
        x = -b / (2 * a)
        return [x]
    else:
        x1 = (-b + D ** 0.5) / (2 * a)
        x2 = (-b - D ** 0.5) / (2 * a)
        return [x1, x2]


def get_y(x, coeffs):
    return sum(x ** i * coeffs[::-1][i] for i in range(len(coeffs)))


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    skew = np.mean((x - np.mean(x)) ** 3) / np.std(x) ** 3
    return round(skew, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    kurtosis = np.mean((x - np.mean(x)) ** 4) / np.std(x) ** 4 - 3
    return round(kurtosis, 2)
