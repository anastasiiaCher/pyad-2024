import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    def matrix_multiplication(matrix1, matrix2):
      m = len(matrix1)
      n = len(matrix1[0])
      n_2 = len(matrix2)
      p = len(matrix2[0])
    
      
      if (n != n_2): 
        raise ValueError()
    
      c = [[0 for x in range(p)] for y in range(m)]
    
      for i in range(m):
        for k in range(p):
          temp_sum = 0
          for j in range(n):
            temp_sum += matrix1[i][j]*matrix2[j][k] 
          c[i][k] = temp_sum
      
      return c

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # put your code here
    eq1 = [int(x) for x in coeffs1.split()]
    eq2 = [int(x) for x in coeffs2.split()]

    if (eq1 == eq2):
      return

    def f(x):
      return eq1[0] * x**2 + eq1[1] * x + eq1[2]
    

    a = (eq1[0]-eq2[0])
    b = (eq1[1]-eq2[1])
    c = (eq1[2]-eq2[2])


    if a != 0:
      discriminant = b**2 - 4*a*c

      if discriminant < 0:
        return []
          
      x1 = (-b + discriminant**0.5) / (2*a)
      x2 = (-b - discriminant**0.5) / (2*a)

      if x1 != x2:
        return [(x1, f(x1)), (x2, f(x2))]
      return [(x1, f(x1))]
    elif a == 0 and b != 0:
      x = -c / b
      return [(x, f(x))]
    else:
      return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    x_e = sum(x)/len(x)
    m_2 = (sum([(x_i - x_e)**2 for x_i in x])/len(x))**0.5
    m_3 = sum([(x_i - x_e)**3 for x_i in x])/len(x)
    A_3 = m_3/(m_2**3)

    return round(A_3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    x_e = sum(x)/len(x)
    m_2 = (sum([(x_i - x_e)**2 for x_i in x])/len(x))**0.5
    m_4 = sum([(x_i - x_e)**4 for x_i in x])/len(x)
    E_4 = m_4/(m_2**4) - 3

    return round(E_4, 2)
