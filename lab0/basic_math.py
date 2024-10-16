import numpy as np

def matrix_multiplication(matrix1, matrix2):
    rows_matrix1 = len(matrix1)
    cols_matrix1 = len(matrix1[0])
    rows_matrix2 = len(matrix2)
    cols_matrix2 = len(matrix2[0])

    if cols_matrix1 != rows_matrix2:
        raise ValueError("Матрицы нельзя перемножить.")

    matrix3 = [[0 for _ in range(cols_matrix2)] for _ in range(rows_matrix1)]

    for i in range(rows_matrix1):
        for j in range(cols_matrix2):
            for k in range(cols_matrix1):
                matrix3[i][j] += matrix1[i][k] * matrix2[k][j]

    return matrix3




def functions(coeffs1, coeffs2):
    a1 = list(map(float, coeffs1.split()))
    a2 = list(map(float, coeffs2.split()))

    a_diff = [a1[i] - a2[i] for i in range(3)]

    if np.all(np.isclose(a_diff, 0)):
        return None

    if a_diff[0] == 0:
        if a_diff[1] == 0:
            return []
        x = -a_diff[2] / a_diff[1]
        return [(round(x, 6), round(a1[0]*x**2 + a1[1]*x + a1[2], 6))]

    discriminant = a_diff[1]**2 - 4*a_diff[0]*a_diff[2]

    if discriminant > 0:
        x1 = (-a_diff[1] + np.sqrt(discriminant)) / (2 * a_diff[0])
        x2 = (-a_diff[1] - np.sqrt(discriminant)) / (2 * a_diff[0])
        return [
            (round(x1, 6), round(a1[0]*x1**2 + a1[1]*x1 + a1[2], 6)),
            (round(x2, 6), round(a1[0]*x2**2 + a1[1]*x2 + a1[2], 6))
        ]
    elif np.isclose(discriminant, 0):
        x = -a_diff[1] / (2 * a_diff[0])
        return [(round(x, 6), round(a1[0]*x**2 + a1[1]*x + a1[2], 6))]
    else:
        return []


def skew(x):
    mean = np.mean(x)
    m2 = np.mean((x - mean) ** 2)
    m3 = np.mean((x - mean) ** 3)
    sigma = np.sqrt(m2)
    skewness = m3 / sigma ** 3
    return round(skewness, 2)


def kurtosis(x):
    mean = np.mean(x)
    m2 = np.mean((x - mean) ** 2)
    m4 = np.mean((x - mean) ** 4)
    sigma = np.sqrt(m2)
    kurt = m4 / sigma ** 4 - 3
    return round(kurt, 2)


