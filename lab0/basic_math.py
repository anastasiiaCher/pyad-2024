import numpy as np
from scipy.optimize import minimize_scalar, fsolve


def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы должно совпадать с числом строк второй.")

    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    a_1 = [float(coef) for coef in a_1.split()]
    a_2 = [float(coef) for coef in a_2.split()]

    # Нахождение общих решений уравнения F(x) = P(x)
    def equation(x):
        return (a_1[0] * x**2 + a_1[1] * x + a_1[2]) - (a_2[0] * x**2 + a_2[1] * x + a_2[2])

    # Если коэффициенты перед x^2, x и свободный член совпадают, у функций бесконечно много решений
    if a_1 == a_2:
        return None

    # Используем fsolve для поиска решений
    solutions = fsolve(equation, [0, 1])  # Начальные приближения: 0 и 1

    # Округляем решения до 2 знаков после запятой
    rounded_solutions = [round(sol, 2) for sol in solutions]

    # Удаляем дублирующиеся решения
    unique_solutions = list(np.unique(rounded_solutions))

    # Проверяем, если решения подходят (иначе пустой список)
    valid_solutions = [sol for sol in unique_solutions if np.isclose(equation(sol), 0, atol=1e-5)]

    # Возвращаем только первое найденное решение, если оно существует
    return valid_solutions[:1] if valid_solutions else []


def skew(x):
    mean = np.mean(x)
    m2 = np.mean((x - mean) ** 2)
    m3 = np.mean((x - mean) ** 3)

    skewness = m3 / (m2 ** 1.5)
    return round(skewness, 2)


def kurtosis(x):
    mean = np.mean(x)
    m2 = np.mean((x - mean) ** 2)
    m4 = np.mean((x - mean) ** 4)

    kurtosis_value = m4 / (m2 ** 2) - 3
    return round(kurtosis_value, 2)