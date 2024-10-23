import numpy as np
from scipy.optimize import minimize_scalar

# Задание №1
def matrix_multiplication(matrix_a, matrix_b):
    # Проверяем возможность умножения: число столбцов matrix1 должно быть равно числу строк matrix2
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы не могут быть перемножены: размерности не соответствуют.")

    result = [[0] * len(matrix_b[0]) for _ in range(len(matrix_a))]

    # Проход по строкам matrix1
    for i in range(len(matrix_a)):
        # Проход по столбцам matrix2
        for k in range(len(matrix_b[0])):
            # Вычисление каждого элемента новой матрицы
            for j in range(len(matrix_b)):
                result[i][k] += matrix_a[i][j] * matrix_b[j][k]

    return result


# Задание №2
def functions(coeffs1, coeffs2):
    # Преобразуем входные строки в коэффициенты
    a11, a12, a13 = map(float, coeffs1.split())
    a21, a22, a23 = map(float, coeffs2.split())

    # Определим функции F(x) и P(x)
    def F(x):
        return a11 * x ** 2 + a12 * x + a13

    def P(x):
        return a21 * x ** 2 + a22 * x + a23

    # Найдем точки экстремума
    res_F = minimize_scalar(F)
    res_P = minimize_scalar(P)

    # Вывод экстремумов
    print(f"Экстремум F(x): x = {res_F.x}, F(x) = {res_F.fun}")
    print(f"Экстремум P(x): x = {res_P.x}, P(x) = {res_P.fun}")

    # Проверим, есть ли общие решения
    # Решаем уравнение (a11 - a21)x^2 + (a12 - a22)x + (a13 - a23) = 0
    A = a11 - a21
    B = a12 - a22
    C = a13 - a23

    if A == 0 and B == 0:
        if C == 0:
            return None  # Бесконечно много решений
        else:
            return []  # Нет решений

    # Если уравнение квадратичное, решаем его
    common_solutions = []
    if A == 0:
        # Линейное уравнение Bx + C = 0
        if B != 0:
            x = -C / B
            common_solutions.append((x, F(x)))
    else:
        # Квадратичное уравнение
        discriminant = B ** 2 - 4 * A * C
        if discriminant >= 0:
            x1 = (-B + np.sqrt(discriminant)) / (2 * A)
            x2 = (-B - np.sqrt(discriminant)) / (2 * A)
            common_solutions.append((x1, F(x1)))
            if discriminant > 0:
                common_solutions.append((x2, F(x2)))

    return sorted(common_solutions)


# Задание №3
def skew(x):
    if len(x) == 0:
        return None  # Возвращаем None для пустой выборки
    mean = sum(x) / len(x)
    m2 = sum((num - mean) ** 2 for num in x) / len(x)  # Момент второго порядка
    m3 = sum((num - mean) ** 3 for num in x) / len(x)  # Момент третьего порядка
    skewness = m3 / (m2 ** (3 / 2)) if m2 != 0 else 0  # Проверка на ноль в знаменателе
    return round(skewness, 2)  # Округляем до 2 знаков после запятой


def kurtosis(x):
    if len(x) == 0:
        return None  # Возвращаем None для пустой выборки
    mean = sum(x) / len(x)
    m2 = sum((num - mean) ** 2 for num in x) / len(x)  # Момент второго порядка
    m4 = sum((num - mean) ** 4 for num in x) / len(x)  # Момент четвертого порядка
    kurt = m4 / (m2 ** 2) - 3 if m2 != 0 else 0  # Проверка на ноль в знаменателе
    return round(kurt, 2)  # Округляем до 2 знаков после запятой