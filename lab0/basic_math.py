import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar
from collections import Counter


def matrix_multiplication(matrix_a, matrix_b):
  rows_a, rows_b, cols_a, cols_b = len(matrix_a), len(matrix_b), len(matrix_a[0]), len(matrix_b[0])
  if cols_a != rows_b:
      return []
  result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
  for i in range(rows_a):
    for k in range(cols_b):
        for j in range(cols_a):
            result[i][k] += matrix_a[i][j] * matrix_b[j][k]

  return result


def quadratic_function(coeffs, x):
    a, b, c = coeffs
    return a * x**2 + b * x + c

def find_extremum_numeric(coeffs):
    result = minimize_scalar(lambda x: quadratic_function(coeffs, x))
    return result.x, quadratic_function(coeffs, result.x)

def functions(a_1, a_2):
    coeffs1 = list(map(float, a_1.strip().split()))
    coeffs2 = list(map(float, a_2.strip().split()))

    x_extremum_1, f_extremum_1 = find_extremum_numeric(coeffs1)
    x_extremum_2, f_extremum_2 = find_extremum_numeric(coeffs2)

    print(f"Экстремум первой функции: x = {x_extremum_1}, значение = {f_extremum_1}")
    print(f"Экстремум второй функции: x = {x_extremum_2}, значение = {f_extremum_2}")

    a, b, c = coeffs1[0] - coeffs2[0], coeffs1[1] - coeffs2[1], coeffs1[2] - coeffs2[2]

    if a == 0 and b == 0 and c==0:
      return None
    elif a == 0:
      if b == 0:
        return []
      ans_x = -c / b
      ans_y = quadratic_function(coeffs1, ans_x)
      return [(ans_x, ans_y)]
    else:
      D = b ** 2 - 4 * a * c
      if D < 0:
        return []
      if D == 0:
        ans_x = -b / (2 *a)
        ans_y = quadratic_function(coeffs1, ans_x)
        return [(ans_x, ans_y)]
      else:
        ans_x1 = (-b + np.sqrt(D)) / (2 * a)
        ans_x2 = (-b - np.sqrt(D)) / (2 * a)
        ans_y1 = quadratic_function(coeffs1, ans_x1)
        ans_y2 = quadratic_function(coeffs1, ans_x2)
        return [(ans_x1, ans_y1), (ans_x2, ans_y2)]


def find_moment(values, power):
  sample_mean = sum(values)/len(values)
  frequency = Counter(values)
  return sum((value - sample_mean)**power * count for value, count in frequency.items()) / len(values)

def find_sigma(values, power):
  sample_mean = sum(values)/len(values)
  dispersion = sum((value - sample_mean)**2 for value in values) / len(values)
  sigma = np.sqrt(dispersion)
  return sigma**power

def skew(x):
  result = find_moment(x, 3)/find_sigma(x, 3)
  return round(result, 2)
    
def kurtosis(x):
  result =  find_moment(x, 4)/find_sigma(x, 4) - 3
  return round(result, 2)
