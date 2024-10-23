import numpy as np
import scipy as sc

def matrix_multiplication(A, B):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(A[0]) != len(B):
        raise ValueError()
    res = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                res[i][j] += A[i][k] * B[k][j]
    return res


def functions(a1, a2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a1 = list(map(float, a1.split()))
    a2 = list(map(float, a2.split()))
    a = a1[0] - a2[0]
    b = a1[1] - a2[1]
    c = a1[2] - a2[2]
    def help(x):
      return a1[0] * x**2 + a1[1] * x + a1[2]
    if a == 0 and b == 0 and c == 0:
        return None
    if a == 0:
        if b == 0:
            return []
        else:
            x = -c / b
            return [(x, help(x))]
    d = np.sqrt(b ** 2 - 4 * a * c)
    if d < 0:
        return []
    elif d == 0:
        x = -b / (2 * a)
        return [(x, help(x))]
    else:
        x1 = (-b + d) / (2 * a)
        x2 = (-b - d) / (2 * a)
        return [(x1, help(x1)), (x2, help(x2))]


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    x_e = sum(x) / len(x)
    m_2 = np.sqrt((sum([(i - x_e) ** 2 for i in x]) / len(x)))
    m_3 = sum([(i - x_e) ** 3 for i in x]) / len(x)
    ans = m_3 / (m_2 ** 3)
    return round(ans, 2)


def kurtosis(x):
    """
    Задание 4. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x_e = sum(x) / len(x)
    m_2 = np.sqrt((sum([(i - x_e) ** 2 for i in x]) / len(x)))
    m_4 = sum([(i - x_e) ** 4 for i in x]) / len(x)
    ans = m_4 / (m_2 ** 4) - 3
    return round(ans, 2)
