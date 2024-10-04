import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # put your code here

    new_matrix = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError() 
    for i in range(len(matrix_a)):
        for j in range(len(matrix_a[0])):
            for k in range(len(matrix_b[0])):
                new_matrix[i][k] += matrix_a[i][j] * matrix_b[j][k]
    return new_matrix


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # put your code here
    eq1 = [int(x) for x in a_1.split()]
    eq2 = [int(x) for x in a_2.split()]

    if eq1 == eq2:
        return

    def f(x):
        return eq1[0]*x**2 + eq1[1]*x + eq1[2]


    #min_result1 = fmin(lambda x: eq1[0]*x**2 + eq1[1]*x + eq1[2], [0]) #точка экстремума первой функции
    #min_result2 = fmin(lambda x: eq2[0]*x**2 + eq2[1]*x + eq2[2], [0]) #точка экстремума второй функции

    #func = (eq1[0]-eq2[0])*x**2 + (eq1[1]-eq2[1])*x + (eq1[2]-eq2[2])

    a = (eq1[0]-eq2[0])
    b = (eq1[1]-eq2[1])
    c = (eq1[2]-eq2[2])

    if a != 0:
        D = b**2 - 4*a*c
        if D < 0:
            return [] 
        else:
            root1 = (-b + D**0.5)/(2*a)
            root2 = (-b - D**0.5)/(2*a)
            if root2 == root1:
                return [(root1, f(root1))]
            else:
                return [(root1, f(root1)), ((root2, f(root2)))]
    elif a == 0 and b != 0:
        root1 = -c/b
        return [(root1, f(root1))]
    else:
        return []




def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    x_e = sum(x)/len(x)
    sigma = (sum([(num - x_e)**2 for num in x])/len(x))**0.5
    m_3 = sum([(num - x_e)**3 for num in x])/len(x)
    A_3 = m_3/(sigma**3)
    return round(A_3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    x_e = sum(x)/len(x)
    sigma = (sum([(num - x_e)**2 for num in x])/len(x))**0.5
    m_4 = sum([(num - x_e)**4 for num in x])/len(x)
    E_4 = m_4/(sigma**4) - 3
    return round(E_4,2)
