import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # put your code here
    result = []
    if len(b) != len(a[0]):
        raise ValueError("Impossible to multiply the matrices")

    for i in range(len(a)):
        result.append([])
        for j in range(len(b[0])):
            c = 0
            for k in range(len(b)):
                c += (a[i][k] * b[k][j])
            result[i].append(c)
    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # put your code here
    result = []
    a_1 = np.array(list(map(float, str(a_1).split())))
    a_2 = np.array(list(map(float, str(a_2).split())))

    coeffs_diff = a_1 - a_2

    a, b, c = coeffs_diff[0], coeffs_diff[1], coeffs_diff[2]

    temp = None

    def f(root, coeffs):
        val = np.polyval(coeffs, root)
        return val

    if np.isclose(a, 0):
        if np.isclose(b, 0):
            if np.isclose(c, 0):
                return temp
            else:
                empty_result = []
                return empty_result
        x = -c / b
        result.append((x, f(x, coeffs_diff)))
        return result

    det = b ** 2 - 4 * a * c
    if det < 0:
        return []

    sqrt_det = np.sqrt(det)
    x1 = (-b + sqrt_det) / (2 * a)
    x2 = (-b - sqrt_det) / (2 * a)

    if np.isclose(x1, x2):
        single_root = [(x1, f(x1, a_1))]
        return single_root
    else:
        result.append((x1, f(x1, a_1)))
        result.append((x2, f(x2, a_1)))
        return result


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    n = len(x)
    mean = sum(x) / n

    variance = sum((xi - mean) ** 2 for xi in x) / n
    st_dev = variance ** 0.5

    a = sum((xi - mean) ** 3 for xi in x) / n
    skeww = a / st_dev ** 3

    return round(skeww, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    n = len(x)
    mean = sum(x) / n
    variance = sum((xi - mean) ** 2 for xi in x) / n

    st_dev = variance ** 0.5
    a = sum((xi - mean) ** 4 for xi in x) / n

    kurtosiss = a / st_dev ** 4 - 3

    return round(kurtosiss, 2)
