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
    pass


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    pass
