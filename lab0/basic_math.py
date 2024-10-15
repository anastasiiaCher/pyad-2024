import numpy as np
import scipy as sc
from scipy.optimize import fsolve


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    #проверка, что умножение возможно
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("умножение невозможно")
    #инициализация результирующей матрицы с нулями
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    #умножение
    for i in range(len(matrix_a)):          
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    #преобразование элементов в int
    result = [[int(element) for element in row] for row in result]
    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    #определим функции
    F = lambda x: a11 * x ** 2 + a12 * x + a13
    P = lambda x: a21 * x ** 2 + a22 * x + a23
    difference = lambda x: F(x) - P(x)

    #корни уравнения F(x)=P(x)
    #решаем уравнение
    A = a11 - a21
    B = a12 - a22
    C = a13 - a23

    if A == 0:
        #если кф при x^2 равен 0, уравнение линейное
        if B == 0:
            if C == 0:
                return None  #бесконечно много решений
            else:
                return []  #нет решений
        else:
            #уравнение имеет одно решение
            root = -C / B
            return [(root, F(root))]
    else:
        #решаем через дискриминант
        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return []  #нет решений
        elif discriminant == 0:
            root = -B / (2 * A)
            return [(root, F(root))]
        else:
            root1 = (-B + np.sqrt(discriminant)) / (2 * A)
            root2 = (-B - np.sqrt(discriminant)) / (2 * A)
            return sorted([(root1, F(root1)), (root2, F(root2))])


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    #выборочное среднее
    mean = sum(x) / n

    #необходимые моменты
    m2 = sum((x - mean) ** 2 for x in x) / n  #дисперсия (м2)
    m3 = sum((x - mean) ** 3 for x in x) / n  #м3

    #стандартное отклонение
    sigma = m2 ** 0.5

    #коэффициент асимметрии
    skewness = m3 / sigma ** 3
    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    #выборочное среднее
    mean = sum(x) / n

    #необходимые моменты
    m2 = sum((x - mean) ** 2 for x in x) / n  #дисперсия (м2)
    m3 = sum((x - mean) ** 3 for x in x) / n  #м3
    m4 = sum((x - mean) ** 4 for x in x) / n  #м4

    #стандартное отклонение
    sigma = m2 ** 0.5

    #коэффициент эксцесса
    kurt = m4 / sigma ** 4 - 3

    return round(kurt, 2)

