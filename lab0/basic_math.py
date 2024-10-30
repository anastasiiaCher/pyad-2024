import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError()
    
    result = [[0 for i in range(len(matrix_b[0]))] for j in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result


def functions(a_1, a_2):
  list_of_coefficient_1 = list(map(float, a_1.split()))
  list_of_coefficient_2 = list(map(float, a_2.split()))

  extremum_1 = sc.minimize_scalar(lambda x: list_of_coefficient_1[0]*x**2 + list_of_coefficient_1[1]*x + list_of_coefficient_1[2])
  extremum_2 = sc.minimize_scalar(lambda x: list_of_coefficient_2[0]*x**2 + list_of_coefficient_2[1]*x + list_of_coefficient_2[2])
  coordinates_of_extremum_1 = (extremum_1.x, extremum_1.fun)
  coordinates_of_extremum_2 = (extremum_2.x, extremum_2.fun)
  
  equation_coefficients = list()
  for i in range(len(list_of_coefficient_1)):
    equation_coefficients.append(list_of_coefficient_1[i] - list_of_coefficient_2[i])
  if all(item == 0 for item in equation_coefficients):
    return None
  if (equation_coefficients[0] == 0) and (equation_coefficients[1] == 0):
    return []
  if equation_coefficients[0] == 0:
    x_1 = - equation_coefficients[2] / equation_coefficients[1]
    y_1 = list_of_coefficient_1[0]*x_1**2 + list_of_coefficient_1[1]*x_1 + list_of_coefficient_1[2]
    return [(x_1, y_1)]
  D = equation_coefficients[1]**2 - 4*equation_coefficients[0]*equation_coefficients[2]
  if D < 0:
    return []
  if D == 0:
    x_1 = -equation_coefficients[1] / (2 * equation_coefficients[0])
    y_1 = list_of_coefficient_1[0]*x_1**2 + list_of_coefficient_1[1]*x_1 + list_of_coefficient_1[2]
    return [(x_1, y_1)]
  x_1 = (-equation_coefficients[1] + np.sqrt(D)) / (2 * equation_coefficients[0])
  x_2 = (-equation_coefficients[1] - np.sqrt(D)) / (2 * equation_coefficients[0])
  y_1 = list_of_coefficient_1[0]*x_1**2 + list_of_coefficient_1[1]*x_1 + list_of_coefficient_1[2]
  y_2 = list_of_coefficient_1[0]*x_2**2 + list_of_coefficient_1[1]*x_2 + list_of_coefficient_1[2]
  return [(x_1, y_1), (x_2, y_2)]


def skew(x):
    n = len(x)
    mean = sum(x) / n
    variance = sum((i - mean) ** 2 for i in x) / n
    std_dev = np.sqrt(variance)
    skew = sum((i - mean) ** 3 for i in x) / (n * std_dev**3)
    
    return round(skew, 2)


def kurtosis(x):
    n = len(x)  
    mean = sum(x) / n
    variance = sum((i - mean) ** 2 for i in x) / n
    std_dev = np.sqrt(variance)
    kurt = sum((i - mean) ** 4 for i in x) / (n * std_dev**4) - 3
    
    return round(kurt, 2)
