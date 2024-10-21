import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    ai, aj = len(matrix_a), len(matrix_a[0])
    bi, bj = len(matrix_b), len(matrix_b[0])
    if aj != bi:
      raise ValueError('Matrices cannot be multiplied. The number of columns in matrix A must be equal to the number of rows in matrix B.')
    r = [[0] * bj for _ in range(ai)]
    for i in range(ai):
      for j in range(bj):
        for k in range(aj):
          r[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return r

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a1, b1, c1 = map(int, a_1.split())
    a2, b2, c2 = map(int, a_2.split())
    if a1 != 0:
      x1 = -b1 / (2 * a1)
      y1 = q_func(x1, a1, b1, c1)
      print('Экстремум 1 функции:', (x1, y1))
    if a2 != 0:
      x2 = -b2 / (2 * a2)
      y2 = q_func(x2, a2, b2, c2)
      print('Экстремум 2 функции:', (x2, y2))
    a = a1 - a2
    b = b1 - b2
    c = c1 - c2
    d = b**2 - 4 * a * c
    if a1 == a2 and b1 == b2 and c1 == c2:
      return None
    elif d < 0 or a1 == a2 and b1 == b2 and c1 != c2:
      return []
    elif d >= 0:
      if a == 0:
        return [(-c / b, 0)]
      x1 = (-b + d**0.5) / (2 * a)
      y1 = q_func(x1, a1, b1, c1)
      if d == 0:
        return [(x1, y1)]
      x2 = (-b - d**0.5) / (2 * a)
      y2 = q_func(x2, a1, b1, c1)
      return [(x1, y1), (x2, y2)]

def q_func(x, a, b, c):
   return a * x**2 + b * x + c

def m3(x):
  return sum((i - sum(x) / len(x))**3 for i in x) / len(x)

def m4(x):
  return sum((i - sum(x) / len(x))**4 for i in x) / len(x)

def sigma(x):
  return (sum((i - sum(x) / len(x))**2 for i in x) / len(x)) ** 0.5

def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    return round(m3(x) / sigma(x)**3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    return round(m4(x) / sigma(x)**4 - 3, 2)
