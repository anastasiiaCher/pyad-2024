import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    matrix_a_rows, matrix_a_columns = np.shape(matrix_a)
    matrix_b_rows, matrix_b_columns = np.shape(matrix_b)
    if matrix_a_columns != matrix_b_rows:
        raise ValueError("Matrixes can not be multiplicated")
    result = [[0] * np.shape(matrix_a)[0] for _ in range(np.shape(matrix_b)[1])]
    for i in range(matrix_a_rows):
        for j in range(matrix_b_columns):
            for s in range(matrix_a_columns):
                result[i][j] += matrix_a[i][s] * matrix_b[s][j]
    return result

def quadratic_equation(x: float, a: list[float]) -> float:
  return a[0] * x**2 + a[1] * x + a[2]

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coeffs1 = list(map(float, a_1.split()))
    coeffs2 = list(map(float, a_2.split()))

    f_extr_result = sc.optimize.minimize_scalar(quadratic_equation, args=(coeffs1))
    p_extr_result = sc.optimize.minimize_scalar(quadratic_equation, args=(coeffs2))

    print(f"Function F(x) estimated point of extremum is x_extr = {f_extr_result.x}. F(x_extr) = {f_extr_result.fun}")
    print(f"Function P(x) estimated point of extremum is x_extr = {p_extr_result.x}. F(x_extr) = {p_extr_result.fun}")

    a, b, c  = [coeffs1[i] - coeffs2[i] for i in [0, 1, 2]]

    d = b ** 2 - 4 * a * c

    if a == 0 and b == 0 and c == 0:
        return None
    elif a == 0 and b == 0:
        return []
    elif a == 0:
        return [(-c/b, quadratic_equation(-c/b, coeffs1))]
    else:
        if d < 0:
            return []
        elif d == 0:
            return[(-b / (2 * a), quadratic_equation(-b / (2 * a), coeffs1))]
        else:
            return[((-b + d**0.5) / (2 * a), quadratic_equation((-b + d**0.5) / (2 * a), coeffs1)),
                    ((-b - d**0.5) / (2 * a), quadratic_equation((-b - d**0.5) / (2 * a), coeffs1))]
        

def sample_mean(sample: np.array) -> float:
  """
  Функция вычисляющая среднее выборочное
  """
  return sum(sample)/len(sample)


def moment(sample: np.array, n: int) -> float:
  """
  Функция вычисляющая n-тый момент
  """
  result = 0
  sm = sample_mean(sample)
  for x in sample:
    result += (x - sm)**n
  return result/len(sample)


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    return round(moment(x, 3)/moment(x, 2)**(3/2), 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    return round(moment(x, 4)/moment(x, 2)**2 - 3, 2)
