import numpy as np
import scipy as sc

def matrix_multiplication(matrix_a, matrix_b):
    # Проверка
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы нельзя перемножить")

    # Создание результирующей матрицы с нулями
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a1, b1, c1 = list(map(int, a_1.split()))
    a2, b2, c2 = list(map(int, a_2.split()))
    def square_eq(a, b, c):
        f = lambda x: a*x**2 + b*x + c
        print(a,b,c)
        if a != 0:
            d = b**2 - 4 * a * c
            if d > 0:
                x1 = round((-b + d**0.5)/(2*a),0)
                x2 = round((-b - d**0.5)/(2*a),0)
            elif d == 0:
                x1 = x2 = round(-b/(2*a), 0)
            else:
                return None
            return (x1, x2)
        else:
            if b != 0:
                x1 = -c/b
                return (x1, x1)
            else:
                return None

    peak1 = peak2 = None
    if a1 != 0:
        peak1 = (-b1/(2*a1), c1-b1**2/(4*a1))
    if a2 != 0:
        peak2 = (-b2/(2*a2), c2-b2**2/(4*a2))

    if a1 != 0 and a2 != 0:
        if peak1 == peak2 and a1 * a2 > 0:
            # Бесконечно много решений
            if a1 == a2:
                return None
        elif peak1 == peak2 and a1 * a2 < 0:
            # Нет решений
            return False
        res = square_eq(a1-a2, b1-b2, c1-c2)
        if res == None:
            return []
        x1, x2 = res
        f = lambda x: a1*x**2 + b1*x + c1
        if x1 == x2:
            return [(x1, f(x1))]
        else:
            return [(x1, f(x1)), (x2, f(x2))]
    elif a1 == 0 and a2 == 0:
        f = lambda x: b1*x + c1
        if b1 == b2 and c1 == c2:
            # Совпадение прямых
            return None
        # Прямые
        x1 = (c2 -c1)/(b1 - b2)
        return [(x1, f(x1))]
    # Одна прямая
    elif a1 == 0 and a2 != 0:
        res = square_eq(a1-a2, b1-b2, c1-c2)
        if res == None:
            return []
        x1, x2 = res
        f = lambda x: a1*x**2 + b1*x + c1
        if x1 == x2:
            return [(x1, f(x1))]
        else:
            return [(x1, f(x1)), (x2, f(x2))]
    elif a1 != 0 and a2 == 0:
        res = square_eq(a1-a2, b1-b2, c1-c2)
        if res == None:
            return []
        x1, x2 = res
        f = lambda x: a1*x**2 + b1*x + c1
        if x1 == x2:
            return [(x1, f(x1))]
        else:
            return [(x1, f(x1)), (x2, f(x2))]
    # (a1-a2)x**2 + (b1-b2)x + (c1-c2)
    # D = (b1-b2)**2 -4(a1-a2)(c1-c2)
    # x1 = (-(b1-b2)+round(D**0.5,2))/2*(a1-a2)
    # extr1 = sc.optimize.minimize_scalar(lambda x: coefs1[0] * x **2 + coefs1[1] * x + coefs1[2])
    # extr2 = sc.optimize.minimize_scalar(lambda x: coefs2[0] * x **2 + coefs2[1] * x + coefs2[2])
    # print(extr1.x, extr2.x)

def momentum(pow, arr, av: float=0):
    summ = 0
    for i in arr:
        summ += (i - av) ** pow
    return summ / len(arr)


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    av = momentum(1, x)
    disp = momentum(2, x, av)
    return round(momentum(3, x, av) / disp**1.5, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    av = momentum(1, x)
    disp = momentum(2, x, av)
    return round(momentum(4, x, av) / disp**2 - 3, 2)
