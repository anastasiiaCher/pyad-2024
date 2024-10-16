import math
from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError

    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += (matrix_a[i][k] * matrix_b[k][j])

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций. Необходимо
    найти точки экстремума функции и определить, есть ли у функций общие решения. Вернуть нужно
    координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    f_list = list(map(int, a_1.split()))
    p_list = list(map(int, a_2.split()))

    def create_fun(x, list_coef):
        return list_coef[0] * x ** 2 + list_coef[1] * x + list_coef[2]

    def find_extremum(list_coef):
        extremum = minimize_scalar(create_fun, args=(list_coef, ), bounds=[-10, 10])
        result_x = round(extremum.x, 2)
        return result_x, extremum.fun

    def find_solution(f_coef, p_coef):
        a = f_coef[0] - p_coef[0]
        b = f_coef[1] - p_coef[1]
        c = f_coef[2] - p_coef[2]

        if a == 0 and b == 0 and c == 0:
            return None

        if a != 0:
            d = b ** 2 - 4 * a * c
            if d < 0:
                return []
            else:
                x1 = (-b + d ** 0.5) / 2 * a
                x2 = (-b - d ** 0.5) / 2 * a
                if x1 == x2:
                    return [x1, create_fun(x1, f_coef)]
                else:
                    return [(x1, create_fun(x1, f_coef)), (x2, create_fun(x2, f_coef))]
        elif b != 0:
            x = -c / b
            return [(x, create_fun(x, [a, b, c]))]

        else:
            return []

    solutions = find_solution(f_list, p_list)

    print(f"Extremum F(x): x = {find_extremum(f_list)[0]}, f(x) = {find_extremum(f_list)[1]}")
    print(f"Extremum P(x): x = {find_extremum(p_list)[0]}, p(x) = {find_extremum(p_list)[1]}")

    return solutions


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """

    x.sort()
    if len(x) < 2:
        raise ValueError

    # Нахождение среднего квадратичного
    xe = sum(x) / len(x)

    # Нахождение стандартного отклонения
    D = sum((x - xe) ** 2 for x in x) / len(x)
    d = math.sqrt(D)

    # Коэффициент асимметрии
    numerator = 0
    for xi in set(x):
        ni = x.count(xi)
        numerator += ((xi - xe) ** 3) * ni
    m3 = numerator / len(x)
    A3 = m3 / (d ** 3)

    return round(A3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """

    x.sort()
    if len(x) < 2:
        raise ValueError

    # Нахождение среднего квадратичного
    xe = sum(x) / len(x)

    # Нахождение стандартного отклонения
    D = sum((x - xe) ** 2 for x in x) / len(x)
    d = math.sqrt(D)

    # Эксцесс выборки
    numerator = 0
    for xi in set(x):
        ni = x.count(xi)
        numerator += ((xi - xe) ** 4) * ni
    m4 = numerator / len(x)
    A4 = (m4 / (d ** 4)) - 3

    return round(A4, 2)
