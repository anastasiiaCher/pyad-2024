import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) == len(matrix_b):
      m = len(matrix_a)
      n = len(matrix_b)
      p = len(matrix_b[0])
      result = [[0] * m for i in range(p)]
      for i in range(m):
        for k in range(p):
          for j in range(n):
            result[i][k] += matrix_a[i][j] * matrix_b[j][k]
      print(*result, sep='\n')
    else:
      print("wrong matrix length")


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    def poly(abc, arg):
        return abc[0] * arg ** 2 + abc[1] * arg + abc[2]

    def res(arg):
        return arg, poly(a_1, arg)

    def extremum(coefficients):
        minimum = minimize_scalar(lambda f: poly(coefficients, f))
        return minimum.x, poly(coefficients, minimum.x)

    a_1 = list(map(int, a_1.split()))
    a_2 = list(map(int, a_2.split()))

    extremum_1 = extremum(a_1)
    extremum_2 = extremum(a_2)

    a, b, c = [a_1[i] - a_2[i] for i in range(3)]
    d = b * b - 4 * a * c

    if a == 0:
        if b == 0:
            if c == 0:
                return None
            return []
        x = -c / b
        return [(res(x))]

    if d == 0:
        x = -b / a / 2
        return [(res(x))]
    elif d > 0:
        d = d ** 0.5
        x1, x2 = (-b + d) / a / 2, (-b - d) / a / 2
        return [(res(x1)), (res(x2))]
    return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)

    xavg = sum(i for i in x) / n

    m2 = sum((i - xavg) ** 2 for i in x) / n
    m3 = sum((i - xavg) ** 3 for i in x) / n
    m4 = sum((i - xavg) ** 4 for i in x) / n

    A3 = m3 / (m2 ** 0.5) ** 3
    print(A3)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)

    xavg = sum(i for i in x) / n

    m2 = sum((i - xavg) ** 2 for i in x) / n
    m3 = sum((i - xavg) ** 3 for i in x) / n
    m4 = sum((i - xavg) ** 4 for i in x) / n

    E4 = m4 / (m2 ** 2) - 3
    print(E4)
