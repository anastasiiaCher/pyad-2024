import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
	rows_a = len(matrix_a)
	cols_a = len(matrix_a[0])
	rows_b = len(matrix_b)
	cols_b = len(matrix_b[0])
    
	if cols_a != rows_b:
		raise ValueError("Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы.")
	else:
		result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
			
		for i in range(rows_a):
			for j in range(cols_b):
				for k in range(cols_a):
					result[i][j] += matrix_a[i][k] * matrix_b[k][j]

		return result


def functions(a_1, a_2):
	coeffs_a = list(map(float, a_1.split()))
	coeffs_b = list(map(float, a_2.split()))
	
	if coeffs_a == coeffs_b:
		return None 
	
	A = coeffs_a[0] - coeffs_b[0]
	B = coeffs_a[1] - coeffs_b[1]
	C = coeffs_a[2] - coeffs_b[2]

	def findY(x):
		return coeffs_a[0] * (x ** 2) + coeffs_a[1] * x + coeffs_a[2]	

	if A == 0:
		if B == 0:
			return None if C == 0 else [] 
		else:
			root = -C / B
			return [(root, findY(root))]
	
	discriminant = B**2 - 4*A*C
	common_solutions = []

	if discriminant > 0:
		root1 = (-B + (discriminant)**0.5) / (2*A)
		root2 = (-B - (discriminant)**0.5) / (2*A)
		common_solutions.extend([(root1, findY(root1)), (root2, findY(root2))])
	elif discriminant == 0:
		root = -B / (2*A)
		common_solutions.append((root, findY(root)))
		
	return common_solutions  


def skew(x):
    n = len(x)
    mean = sum(x) / n
    std_dev = (sum((i - mean) ** 2 for i in x) / n) ** 0.5
    skewness = (sum((i - mean) ** 3 for i in x) / n) / (std_dev ** 3)
    return round(skewness, 2)


def kurtosis(x):
    n = len(x)
    mean = sum(x) / n
    std_dev = (sum((i - mean) ** 2 for i in x) / n) 
    kurt = (sum((i - mean) ** 4 for i in x) / n) / (std_dev ** 2) - 3 
    return round(kurt, 2)

