import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число стобцов матрицы A не равно числу строк матрицы B.")

    common_dim = len(matrix_a[0])

    result_matrix = [[None for col in matrix_b[0]] for row in matrix_a]

    for i in range(len(result_matrix)):
        for j in range(len(result_matrix[0])):
            cell_result = 0
            for k in range(common_dim):
                cell_result += matrix_a[i][k] * matrix_b[k][j]
            result_matrix[i][j] = cell_result

    return result_matrix


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    def get_params(params_str: str):
        return [float(p) for p in params_str.split()]

    def get_func(params: list[float]):
        def f(x):
            return params[0] * x**2 + params[1] * x + params[2]

        return f

    def get_extremum_point(params_str: str):
        params = get_params(params_str)
        if params[0] < 0:
            params = [-p for p in params]
        return sc.optimize.minimize_scalar(get_func(params)).x

    a1 = get_params(a_1)
    a2 = get_params(a_2)
    params_difference = [a1[i] - a2[i] for i in range(len(a1))]

    if params_difference == [0, 0, 0]:
        return None  # функции полностью совпадают, общих точек бесконечно много

    params_difference.reverse()
    p = np.polynomial.Polynomial(params_difference)
    intersection_x = p.roots()

    if len(intersection_x) == 0:
        return []  # общих точек нет
    if intersection_x[0].imag != 0:
        return []  # есть комплексный корень не подходит

    f_1 = get_func(a1)
    intersection_y = [f_1(x) for x in intersection_x]

    return list(zip(map(float, intersection_x), map(float, intersection_y)))


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m_3 = np.mean((x - np.mean(x)) ** 3)
    s = np.sqrt(np.var(x))
    skew = m_3 / s**3
    return np.round(skew, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m_4 = np.mean((x - np.mean(x)) ** 4)
    s = np.sqrt(np.var(x))
    kurtosis = (m_4 / s**4) - 3
    return np.round(kurtosis, 2)
