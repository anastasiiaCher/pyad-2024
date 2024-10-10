import numpy as np
from scipy.optimize import minimize_scalar, fmin, root, fsolve
from collections import Counter


def matrix_multiplication(matrix_a, matrix_b):
	"""
	Задание 1. Функция для перемножения матриц с помощью списков и циклов.
	Вернуть нужно матрицу в формате списка.
	"""
	output_row = []
	output_matrix = []
	matrix_a = np.array(matrix_a)
	matrix_b = np.array(matrix_b)
	number_of_rows = matrix_a.shape[0]
	number_of_columns = matrix_b.shape[1]
	#print(number_of_rows, number_of_columns)
	if matrix_a.shape[1] == matrix_b.shape[0]:
		for i in range(number_of_rows):
		  row = matrix_a[i]
		  output_row = []
		  for j in range(number_of_columns):
		    column = np.array([a[j] for a in matrix_b])
		    output_row.append(sum(row*column))
		  #print(output_row)
		  output_matrix.append(output_row)
		return output_matrix
	else:
		raise ValueError


def functions(a_1, a_2):
	"""
	Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
	Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
	Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
	"""
	# put your code here
	coefs_1 = [float(a) for a in a_1.split()]
	coefs_2 = [float(a) for a in a_2.split()]
	def f1(x):
	  return coefs_1[0]*x**2 + coefs_1[1]*x + coefs_1[2]
	def f2(x):
	  return coefs_2[0]*x**2 + coefs_2[1]*x + coefs_2[2]
	def get_extr(func, coefs):
	  if coefs[0] > 0:
	    return f'Экстремум в точке: {minimize_scalar(func).x}'
	#print("----Функция1----\n", get_extr(f1, coefs_1))
	#print("----Функция2----\n", get_extr(f2, coefs_2))
	def system(x):
	  return f1(x)-f2(x)
	x = np.linspace(-4, 4, 100)
	res = root(system, x)
	xroots = np.unique(res.x.round())
	yroots = f2(xroots)
	sols = []
	#print(res)
	# status 5: The iteration is not making good progress, as measured by the improvement from the last ten iterations.
	if res.success:
	  for i in range(len(xroots)):
	    sols.append((round(xroots[i]), round(yroots[i])))
	    # Так как у квадратного уравнения всего 2 корня, то если корней больше 2, значит корней бесконечно много
	    # (если будет 2 неквадратных уравнения(coeffs[0] = 0) - тем более так как у линейных максимум 1 общее решение или бесконечно, если они совпадают)
	    if len(sols) > 2:
	      return None
	return sols


def skew(x):
	"""
	Задание 3. Функция для расчета коэффициента асимметрии.
	Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
	"""
	# put your code here
	cnt = Counter(x)
	n = len(x)
	m = sum(x)/n
	#print(cnt.items())
	# дисперсия - сигма квадрат
	m2 = sum([(xi-m)**2*ni for (xi, ni) in cnt.items()])/n
	m3 = sum([(xi-m)**3*ni for (xi, ni) in cnt.items()])/n
	m4 = sum([(xi-m)**4*ni for (xi, ni) in cnt.items()])/n
	A3 = m3/(m2**(3/2))
	return round(A3, 2)


def kurtosis(x):
	"""
	Задание 3. Функция для расчета коэффициента эксцесса.
	Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
	"""
	# put your code here
	cnt = Counter(x)
	n = len(x)
	m = sum(x)/n
	#print(cnt.items())
	m2 = sum([(xi-m)**2*ni for (xi, ni) in cnt.items()])/n
	m3 = sum([(xi-m)**3*ni for (xi, ni) in cnt.items()])/n
	m4 = sum([(xi-m)**4*ni for (xi, ni) in cnt.items()])/n
	E4 = m4/(m2**2) - 3
	return round(E4, 2)
