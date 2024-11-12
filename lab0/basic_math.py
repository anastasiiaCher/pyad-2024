import math
from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы НЕ равно числу строк второй матрицы!")
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
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
    coefficients_f, coefficients_p = list(map(float, a_1.split())), list(map(float, a_2.split()))
    if coefficients_f == coefficients_p:
        return None

    F = lambda x: coefficients_f[0] * x ** 2 + coefficients_f[1] * x + coefficients_f[2]
    P = lambda x: coefficients_p[0] * x ** 2 + coefficients_p[1] * x + coefficients_p[2]
    extremum_f, extremum_p = minimize_scalar(F), minimize_scalar(P)

    result = []
    if extremum_f.success and extremum_p.success:
        if math.isclose(extremum_f.x, extremum_p.x) and math.isclose(extremum_f.fun, extremum_p.fun):
            result.append((extremum_f.x, extremum_f.fun))

    # fx = px для нахождения точек пересечения
    a = coefficients_f[0] - coefficients_p[0]
    b = coefficients_f[1] - coefficients_p[1]
    c = coefficients_f[2] - coefficients_p[2]
    if a == 0:
        if b == 0:
            if c == 0:
                return None
            return []
        sol = -c / b
        if math.isclose(F(sol), P(sol)):
            result.append((sol, F(sol)))
    else:
        discriminant = b ** 2 - 4 * a * c

        if discriminant > 0:
            sol1 = (-b + discriminant ** 0.5) / (2 * a)
            sol2 = (-b - discriminant ** 0.5) / (2 * a)
            for sol in (sol1, sol2):
                if math.isclose(F(sol), P(sol)):
                    result.append((sol, F(sol)))

        elif discriminant == 0:
            sol = -b / (2 * a)
            if math.isclose(F(sol), P(sol)):
                result.append((sol, F(sol)))

    return result


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    mean = sum(x) / len(x)
    m2 = sum((xi - mean) ** 2 for xi in x) / len(x)
    m3 = sum((xi - mean) ** 3 for xi in x) / len(x)
    sigma = m2 ** 0.5
    A_3 = m3 / sigma ** 3
    return round(A_3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    mean = sum(x) / len(x)
    m2 = sum((xi - mean) ** 2 for xi in x) / len(x)
    m4 = sum((xi - mean) ** 4 for xi in x) / len(x)
    sigma = m2 ** 0.5
    E_4 = (m4 / sigma ** 4) - 3
    return round(E_4, 2)
