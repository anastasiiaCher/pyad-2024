import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    m = len(matrix1)
    n = len(matrix2)
    p = len(matrix2[0])
    c = [[None for d in range(p)] for d in range(m)]

    if len(matrix1[0]) == len(matrix2):
      for i in range(m):
        for j in range(p):
            c[i][j] = sum(matrix1[i][f] * matrix2[f][j] for f in range(n))
      print ("Произведенеие матриц:", c)
    else:
      print ("Матрицы невозможно перемножить")
    pass


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    def F(x, a):
        return a[0] * x**2 + a[1] * x + a[2]
    def P(x, b):
        return b[0] * x**2 + b[1] * x + b[2]
    def find_extremum(F, a):
        x_ext = -a[1] / (2 * a[0])
        extremum_value = F(x_ext, a)
        return x_ext, extremum_value
    def find_common_solutions(a, b):
        coeffs = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        if coeffs[0] == 0 and coeffs[1] == 0 and coeffs[2] == 0:
            return "Бесконечно много решений"
        if coeffs[0] == 0:
            if coeffs[1] == 0:
                return "Нет решений"
            return f"Одно решение: x = {-coeffs[2] / coeffs[1]}"
        D = coeffs[1]**2 - 4 * coeffs[0] * coeffs[2]
        if D < 0:
            return "Нет решений"
        elif D == 0:
            x = -coeffs[1] / (2 * coeffs[0])
            return f"Одно решение: x = {x}"
        else:
            x1 = (-coeffs[1] + D**0.5) / (2 * coeffs[0])
            x2 = (-coeffs[1] - D**0.5) / (2 * coeffs[0])
            return f"Два решения: x1 = {x1}, x2 = {x2}"
    
    line1 = input("Введите коэффициенты для F(x): ")
    line2 = input("Введите коэффициенты для P(x): ")
    extremum_F = find_extremum(F, a)
    extremum_P = find_extremum(P, b)
    common_solutions = find_common_solutions(a, b)
    print(f"Экстремум F(x) в точке x = {extremum_F[0]}, значение F(x) = {extremum_F[1]}")
    print(f"Экстремум P(x) в точке x = {extremum_P[0]}, значение P(x) = {extremum_P[1]}")
    print(f'Общие решения:{common_solutions}')
    pass


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    sr = sum(m) / len(m)
    def disp(m, mean, x):
      mom = 0
      for j in range(len(m)):
        mom += (m[j]-mean) ** x
      mom = mom/len(m)
      return mom
    
    def res(m, m2,m3,m4):
      A3 = m3/(m2**3**0.5)
      E4 = m4/(m2**4**0.5)-3
      return A3
    
    m2 = disp(m, sr, 2)
    m3 = disp(m, sr, 3)
    m4 = disp(m, sr, 4)
    res = res(m, m2, m3, m4)
    print(f'Среднее {sr}\n m2 {m2}\n m3 {m3}\n m4 {m4}\n Коэффициенты асимметрии и эксцесса: {res}')

    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    sr = sum(m) / len(m)
    def disp(m, mean, x):
      mom = 0
      for j in range(len(m)):
        mom += (m[j]-mean) ** x
      mom = mom/len(m)
      return mom
    
    def res(m, m2,m3,m4):
      A3 = m3/(m2**3**0.5)
      E4 = m4/(m2**4**0.5)-3
      return E4
    
    m2 = disp(m, sr, 2)
    m3 = disp(m, sr, 3)
    m4 = disp(m, sr, 4)
    res = res(m, m2, m3, m4)
    print(f'Среднее {sr}\n m2 {m2}\n m3 {m3}\n m4 {m4}\n Коэффициенты асимметрии и эксцесса: {res}')
    pass
