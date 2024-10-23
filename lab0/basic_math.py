import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    #проверим, что умножение выполнить возможно
    m = len(matrix_a)
    n = len(matrix_a[0])
    p = len(matrix_b[0])
    # print(m, n, p, len(matrix_b))
    if (n != len(matrix_b)):
      print("Умножение выполнить не возможно")
    else:
      matrix_result = [[0 for i in range(n)] for j in range(m)]

      for i in range(m):
        for j in range(n):
          matrix_result[i][j] = sum(matrix_a[i][k] * matrix_b[k][j] for k in range(p))
    return matrix_result
    pass

def F(x, a11, a12, a13, flag):
  if flag:
    return a11 * x**2 + a12 * x + a13
  else:
    return -a11 * x**2 - a12 * x - a13

def P(x, a21, a22, a23, flag):
  if flag:
    return a21 * x**2 + a22 * x + a23
  else:
    return -a21 * x**2 - a22 * x - a23

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    result = []

    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    #Парабола с осями вверх
    if a11 > 0:
      result.append(minimize_scalar(F, args=(a11, a12, a13, True)).fun)
    if a21 > 0:
      result.append(minimize_scalar(P, args=(a21, a22, a23, True)).fun)
    #Парабола с осями вниз
    if a11 < 0:
      result.append(-minimize_scalar(F, args=(a11, a12, a13, False)).fun)
    if a21 < 0:
      result.append(-minimize_scalar(P, args=(a21, a22, a23, False)).fun)
    #Прямая
    if a11 == 0:
      result.append(None)
    if a21 == 0:
      result.append(None)

    a = a11 - a21
    b = a12 - a22
    c = a13 - a23
    if a == 0 and b == 0 and c == 0:
      return result
    else:
      D = b**2 - 4*a*c
      if D > 0:
        x1 = (-b + math.sqrt(D)) / (2 * a)
        x2 = (-b - math.sqrt(D)) / (2 * a)
        result.extend([x1, x2])

      elif D == 0:
        x = -b / (2*a)
        result.append(x)
        
      # else:
      #   return []
      return result
    pass


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(sample)
    mean_sample = np.mean(sample)
    
    m2 = np.sum((sample - mean_sample) ** 2) / n
    m3 = np.sum((sample - mean_sample) ** 3) / n
      
    sigma = np.sqrt(m2)
      
    A3 = m3 / sigma ** 3
    return round(A3, 2)
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean_sample = np.mean(x)
    
    m2 = np.sum((x - mean_sample) ** 2) / n
    m4 = np.sum((x - mean_sample) ** 4) / n
      
    sigma = np.sqrt(m2)
      
    E4 = (m4 / sigma**4) - 3
    return round(E4, 2)
    pass s
