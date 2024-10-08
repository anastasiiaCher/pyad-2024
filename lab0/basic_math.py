import numpy as np
import scipy as sc


def matrix_multiplication(x, y):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    m = len(x)
    n = len(x[0])
    p = len(y[0])
    if len(y) != n:

        raise ValueError("Количество столбцов первой матрицы должно совпадать с количеством строк второй матрицы.")

    C = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += x[i][k] * y[k][j]

    return C


def functions(a_1: str, a_2: str):
    """
     Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком,
    если они есть. None, если их бесконечно много.
    """

    a1, b1, c1 = map(float, a_1.split())
    a2, b2, c2 = map(float, a_2.split())

    def find_extremum(a, b, c):
        if a == 0:
            return None
        x_extremum = -b / (2 * a)
        f_extremum = a * x_extremum ** 2 + b * x_extremum + c
        return (x_extremum, f_extremum)

    def solve_quadratic(a1, b1, c1, a2, b2, c2):
        A = a1 - a2
        B = b1 - b2
        C = c1 - c2

        if A == 0:
            if B == 0:
                if C == 0:
                    return None
                else:
                    return []
            else:
                root = -C / B
                return [(root, a1 * root ** 2 + b1 * root + c1)]

        discriminant = B ** 2 - 4 * A * C
        if discriminant > 0:
            root1 = (-B + np.sqrt(discriminant)) / (2 * A)
            root2 = (-B - np.sqrt(discriminant)) / (2 * A)
            return [(root1, a1 * root1 ** 2 + b1 * root1 + c1), (root2, a1 * root2 ** 2 + b1 * root2 + c1)]
        elif discriminant == 0:
            root = -B / (2 * A)
            return [(root, a1 * root ** 2 + b1 * root + c1)]
        else:
            return []


    common_solutions = solve_quadratic(a1, b1, c1, a2, b2, c2)

    if common_solutions is None:
        return None


    return [(round(root, 5), round(value, 5)) for root, value in common_solutions]
def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    mean = sum(x) / len(x)
    D = sum(list(map(lambda val: val**2, x))) / len(x) - mean**2
    m3 = sum((val - mean) ** 3 for val in x) / len(x)
    return round(m3 / D**(3/2),2)



def kurtosis(x):

    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """

    mean = sum(x) / len(x)
    D = sum(list(map(lambda val: val**2, x))) / len(x) - mean**2
    m4 = sum((i - mean)**4  for i in x)/len(x)
    return round((m4/D**2) - 3,2)