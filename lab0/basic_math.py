import numpy as np
from scipy.optimize import fmin

def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы.")

    result_matrix = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for k in range(len(matrix_b[0])):
            for j in range(len(matrix_b)):
                result_matrix[i][k] += matrix_a[i][j] * matrix_b[j][k]

    return result_matrix


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    def F(x, a11, a12, a13):
        return a11 * x ** 2 + a12 * x + a13

    def P(x, a21, a22, a23):
        return a21 * x ** 2 + a22 * x + a23

    a = list(map(float, a_1.split()))
    b = list(map(float, a_2.split()))

    if a == b:
        return None  # бесконечно много решений

    extr_fx = fmin(F, 0, args=(a[0], a[1], a[2]), disp=False)[0]
    extr_fy = F(extr_fx, *a)
    extr_f = (round(extr_fx, 2), round(extr_fy, 2))

    extr_px= fmin(P, 0, args=(b[0], b[1], b[2]), disp=False)[0]
    extr_py = P(extr_px, *b)
    extr_p = (round(extr_px, 2), round(extr_py, 2))

    # Проверяем совпадение экстремумов по x и y
    common_extrema = []
    if extr_f == extr_p:
        common_extrema.append(extr_f)

    # Находим точки пересечения функций
    coeffs_sub = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    # Вычисляем дискриминант
    A, B, C = coeffs_sub
    disc = B ** 2 - 4 * A * C

    # Проверка на наличие корней
    if A == 0:
        if B == 0:
            return [] 
        root = -C / B
        y_value = round(F(root, *a), 2)
        common_extrema.append((round(root, 2), y_value))
    elif disc < 0:
        return []  # Нет общих решений
    elif disc == 0:
        root = -B / (2 * A)
        y_value = round(F(root, *a), 2)
        common_extrema.append((round(root, 2), y_value))
    else:
        root1 = (-B + np.sqrt(disc)) / (2 * A)
        root2 = (-B - np.sqrt(disc)) / (2 * A)
        common_extrema.append((round(root1, 2), round(F(root1, *a), 2)))
        common_extrema.append((round(root2, 2), round(F(root2, *a), 2)))

    common_extrema = sorted(set(common_extrema))

    return common_extrema if common_extrema else []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    x = np.array(x)
    x_mean = np.mean(x)

    m2 = np.mean((x - x_mean) ** 2)
    m3 = np.mean((x - x_mean) ** 3)
    st_deviation = np.sqrt(m2)

    a3 = m3 / st_deviation ** 3

    return round(a3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x = np.array(x)

    x_mean = np.mean(x)
    m2 = np.mean((x - x_mean) ** 2)
    st_deviation = np.sqrt(m2)

    n = len(x)
    excess_kurtosis = (np.sum((x - x_mean) ** 4) / n) / (st_deviation ** 4) - 3

    return round(excess_kurtosis, 2)