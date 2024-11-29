import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    np_matrix_a = np.array(matrix_a)
    np_matrix_b = np.array(matrix_b)

    if (np_matrix_a.ndim != 2 and np_matrix_b.ndim != 2):
        raise ValueError("Аргументы должны быть двумерными матрицами")

    if (np_matrix_a.shape[1] != np_matrix_b.shape[0]):
        raise ValueError("Размеры матриц не позволяют перемножить")

    m, common, n = *np_matrix_a.shape, np_matrix_b.shape[1]
    res = np.zeros((m, n), dtype=np.int16)

    for i in range(m):
        for j in range(n):
            for k in range(common):
                res[i, j] += np_matrix_a[i, k] * np_matrix_b[k, j]

    return res.tolist()


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # Извлечь коэф-ты ур-ний
    factors1 = np.array([int(c) for c in a_1.split(' ')])
    factors2 = np.array([int(c) for c in a_2.split(' ')])

    # Найти экстремумы
    extrem1 = extrem(*factors1)
    extrem2 = extrem(*factors2)
    
    # Найти точки пересечения
    factors = factors1 - factors2
    roots = solve_equation(*factors)
    if roots == None:
        return None

    func = to_polinomial(factors1)
    return [(x, func(x)) for x in roots]

def to_polinomial(factors):
    n = len(factors)
    return lambda x: sum(x**i * factors[::-1][i] for i in range(n))

def extrem(a, b, c):
    if a == 0:
        return None
    func = to_polinomial([a, b, c])
    x = -b / (2*a)
    return (x, func(x))

def solve_equation(a, b, c):
    if (a == 0):
        if (b == 0):
            # Корней бесконечно много или нет 
            return None if c == 0 else []
        else:
            # Линейное ур-ние
            return [-c / b]

    # Дискриминант
    discr = b**2 - 4*a*c

    print(f"{a}x^2 + {b}x + {c}")
    print(f"Discr = {discr}")

    if (discr < 0):
        # Корней нет
        return []
    elif (discr == 0):
        # Корни равны
        x = -b / (2*a)
        return [x]
    else:
        # Разные корни
        x1 = (-b + discr**(0.5)) / (2*a)
        x2 = (-b - discr**(0.5)) / (2*a)
        return [x1, x2]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    return np.round(m(x, 3) / (m(x, 2) ** (3/2)), 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    return np.round(m(x, 4) / (m(x, 2) ** 2) - 3, 2)


def m(x, factor):
    nums = np.array(x)
    return np.average((nums - np.average(nums))**factor)
