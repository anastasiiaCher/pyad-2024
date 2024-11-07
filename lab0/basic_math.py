import numpy as np


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    try:
        if len(matrix_a[0]) != len(matrix_b):
            raise ValueError("Невозможно сделать. проверьте размерности матриц")
        result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

        for i in range(len(matrix_a)):
            for j in range(len(matrix_b[0])):
                for k in range(len(matrix_b)):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]

        return result
    except Exception as e:
        raise e


def functions(a_1, a_2):
    """
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    try:
        def find_extremum(coefs):
            a, b, c = coefs
            return None if a == 0 and b == 0 else -b / (2 * a) if a != 0 else -c / b

        def evaluate_function(coefs, x):
            a, b, c = coefs
            return a * x ** 2 + b * x + c

        def find_common_solutions(a1, a2):
            a, b, c = a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]
            if a == 0 and b == 0 and c == 0:
                return None
            if a == 0:
                return [] if b == 0 else [(-c / b, evaluate_function(a1, -c / b))]
            discriminant = b ** 2 - 4 * a * c
            if discriminant < 0:
                return []
            x1 = (-b + np.sqrt(discriminant)) / (2 * a)
            x2 = (-b - np.sqrt(discriminant)) / (2 * a)
            return [(x1, evaluate_function(a1, x1)), (x2, evaluate_function(a1, x2))] if discriminant > 0 else [
                (-b / (2 * a), evaluate_function(a1, -b / (2 * a)))]

        coef_1 = list(map(float, a_1.split()))
        coef_2 = list(map(float, a_2.split()))
        extremum_1 = find_extremum(coef_1)
        extremum_2 = find_extremum(coef_2)
        return find_common_solutions(coef_1, coef_2)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def central_moment(x, order):
    """
    Дополнительная функция для расчета центрального момента заданного порядка.
    """
    try:
        n = len(x)
        mean = np.mean(x)
        return np.sum((x - mean) ** order) / n
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    try:
        m2 = central_moment(x, 2)  # 2 цм
        m3 = central_moment(x, 3)  # 3 цм
        skewness = m3 / (m2 ** (3 / 2))
        return round(skewness, 2)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    try:
        m2 = central_moment(x, 2)  # 2 цм
        m4 = central_moment(x, 4)  # 4 цм
        kurtosis_value = m4 / (m2 ** 2) - 3
        return round(kurtosis_value, 2)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None