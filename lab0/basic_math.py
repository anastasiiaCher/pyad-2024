import numpy as np
import scipy as sc

def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix")
    
    result=[[0 for _ in matrix_b[0]] for _ in matrix_a]
    for i in range(len(result)):
        for j in range(len(result[0])):
            for k in range(len(matrix_a[0])):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    if a_1 == a_2:
        return None
    
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())
    a=a11-a21
    b=a12-a22
    c=a13-a23

    def f(a, b, c, x):
        return a * x**2 + b * x + c

    def extremum(a, b):
        if a!=0:
            return -b/(2*a)
        else:
            return "No extremum"

    def D(a, b, c):
        return b**2 - 4*a*c
    
    def x1(a, b, c):
        return (-b + (D(a, b, c))**0.5)/(2*a)
    
    def x2(a, b, c):
        return (-b - (D(a, b, c))**0.5)/(2*a)
    
    ext1=extremum(a11, a12)
    ext2=extremum(a21, a22)

    d=D(a, b, c)

    if a!=0:
        if d<0:
            return []
        elif d==0:
            x=x1(a, b, c)
            return [(x, f(a11, a12, a13, x))]
        else:
            x1=x1(a, b, c)
            x2=x2(a, b, c)
            return [(x1, f(a11, a12, a13, x1)), (x2, f(a11, a12, a13, x2))]
    elif b!=0:
        x=-c/b
        return [(x, f(a11, a12, a13, x))]
    else:
        return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m3=0
    m2=0
    x_mean = sum(x) / len(x)
    for i in x:
        m3 += (i - x_mean) ** 3
        m2 += (i - x_mean) ** 2
    m3 = m3 / len(x)
    m2 = m2 / len(x)
    asymmetry_coefficient = m3 / (m2 ** 1.5)
    return round(asymmetry_coefficient, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m4=0
    m2=0
    x_mean = sum(x) / len(x)
    for i in x:
        m4 += (i - x_mean) ** 4
        m2 += (i - x_mean) ** 2
    m4 = m4 / len(x)
    m2 = m2 / len(x)
    coef=m4/(m2**2)-3
    return round(coef, 2)

