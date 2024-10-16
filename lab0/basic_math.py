import numpy as np
import scipy as sc
import math

def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    Arows, Acols = np.array(matrix_a).shape
    Brows, Bcols = np.array(matrix_b).shape
    if Acols != Brows:
        raise ValueError

    matrix_c = np.zeros((Arows, Bcols))
    for i in range(Arows):
        for j in range(Bcols):
            for r in range(Acols):
                matrix_c[i][j] += matrix_a[i][r] * matrix_b[r][j]
    return matrix_c.tolist()


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    def find_extremum(a, b, c):
        if a == 0:
            return None
        return -b/(2*a)

    a11, a12, a13 = list(map(int, a_1.split())) 
    a21, a22, a23 = list(map(int, a_2.split()))
    func1_extremum = find_extremum(a11, a12, a13)
    func2_extremum = find_extremum(a21, a22, a23)
    a = a11 - a21
    b = a12 - a22
    c = a13 - a23
    if a == 0 and b == 0 and c==0:
        return None
    if a == 0 and b == 0:
       return []
    if a == 0:
       x = -c/b
       return [(x, a11*x**2 + a12*x + a13)]
    D = b*b - 4*a*c
    if D > 0:
        x1 = (-b+math.sqrt(D))/(2*a)
        x2 = (-b-math.sqrt(D))/(2*a)
        return [(x1, a11*x1**2 + a12*x1 + a13), (x2, a11*x2**2 + a12*x2 + a13)]
    if D == 0:
        x = -b/(2*a)
        return [(x, a11*x**2 + a12*x + a13)]
    return []

def calc_moments(sample):
  n = len(sample)
  if n==0:
    return None, None, None

  values = set(sample)
  frequencies = {value: sample.count(value) for value in values}
  xe = 0
  for xi in values:
    ni = frequencies.get(xi,0)
    xe += xi*ni/n

  m2 = 0
  m3 = 0
  m4 = 0
  for xi in values:
    ni = frequencies.get(xi,0)
    m2 += (xi-xe)**2*ni/n
    m3 += (xi-xe)**3*ni/n
    m4 += (xi-xe)**4*ni/n

  return m2, m3, m4

def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m2, m3, m4 = calc_moments(x)
    if m2 == 0 or m2 == None:
       return math.nan
    sd = math.sqrt(m2)
    return round(m3 / sd**3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m2, m3, m4 = calc_moments(x)
    if m2 == 0 or m2 == None:
       return math.nan
    sd = math.sqrt(m2)
    return round(m4 / sd**4 - 3, 2)
