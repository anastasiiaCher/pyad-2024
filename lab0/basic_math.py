import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise(ValueError)
    
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(result)):
        for j in range(len(result[0])):
            s = 0
            for k in range(len(matrix_a[0])):
                s += matrix_a[i][k] * matrix_b[k][j]
            result[i][j] = s
    return result

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coef1 = list(map(int, a_1.split()))
    coef2 = list(map(int, a_2.split()))

    #ищем точки экстремумов
    if coef1[0] != 0:
        point1 = - coef1[1] / (2 * coef1[0])
    if coef2[0] != 0:
        point2 = - coef2[1] / (2 * coef2[0]) 

    #ищем общие точки
    ans = []
    #если совпадают
    if coef1[0] == coef2[0] and coef1[1] == coef2[1] and coef1[2] == coef2[2]:
        return None
    
    #если одна точка
    if coef1[0] == coef2[0]:
        if coef1[1] != coef2[1]:
            x = (coef2[2] - coef1[2]) / (coef1[1] - coef2[1])
            ans.append((x, coef1[0] * x**2 + coef1[1] * x + coef1[2]))
    
    #остальные случаи
    a = coef1[0] - coef2[0]
    b = coef1[1] - coef2[1]
    c = coef1[2] - coef2[2]
    if b**2 - 4*a*c >= 0 and a != 0:
        x1 = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)
        x2 = (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
        if x1 == x2:
            ans.append((x1, coef1[0] * x1**2 + coef1[1] * x1 + coef1[2]))
        else:
            ans.append((x1, coef1[0] * x1**2 + coef1[1] * x1 + coef1[2]))
            ans.append((x2, coef1[0] * x2**2 + coef1[1] * x2 + coef1[2]))
    
    return ans


def skew(x):
    vib_sr = sum(x) / len(x) 
    chisl_m2 = 0
    chisl_m3 = 0
    for i in x:
        chisl_m2 += (i - vib_sr) ** 2
        chisl_m3 += (i - vib_sr) ** 3
        
    m2 = chisl_m2 / len(x)
    m3 = chisl_m3 / len(x)

    return round(m3 / ((m2 ** 0.5) ** 3), 2)




def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    vib_sr = sum(x) / len(x) 
    chisl_m2 = 0
    chisl_m4 = 0
    for i in x:
        chisl_m2 += (i - vib_sr) ** 2
        chisl_m4 += (i - vib_sr) ** 4
        
    m2 = chisl_m2 / len(x)
    m3 = chisl_m4 / len(x)

    return round((m3 / ((m2 ** 0.5) ** 4)) - 3 , 2)
