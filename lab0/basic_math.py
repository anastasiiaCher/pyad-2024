import numpy as np

def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    # put your code here
    # Вычисляем размерность
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0]) if rows_a > 0 else 0
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0]) if rows_b > 0 else 0

    # Проверяем, можно ли перемножить матрицы
    if cols_a != rows_b:
        raise ValueError("Количество столбцов и строк не совпадают")

    # Заполняем результирующую матрицу нулями
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def parse_coefficients(input_str):
    return list(map(float, input_str.split()))


def F(x, a):
    return round(a[0] * x ** 2 + a[1] * x + a[2])


def P(x, b):
    return round(b[0] * x ** 2 + b[1] * x + b[2])


def common_solutions(a, b):
    coeff_diff = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    if np.all(np.isclose(coeff_diff, 0)):  # Если функции совпадают, то бесконечно много решений
        return None

    if coeff_diff[0] == 0:  # Если уравнение линейное
        if coeff_diff[1] == 0:
            return []  # Нет решений
        else:
            x = -coeff_diff[2] / coeff_diff[1]
            return [(round(x), F(round(x), a))]

    discriminant = coeff_diff[1] ** 2 - 4 * coeff_diff[0] * coeff_diff[2]

    if discriminant < 0:
        return []
    elif discriminant == 0:
        x = -coeff_diff[1] / (2 * coeff_diff[0])
        return [(round(x), F(round(x), a))]
    else:
        sqrt_d = np.sqrt(discriminant)
        x1 = (-coeff_diff[1] + sqrt_d) / (2 * coeff_diff[0])
        x2 = (-coeff_diff[1] - sqrt_d) / (2 * coeff_diff[0])
        return [(round(x1), F(round(x1), a)), (round(x2), F(round(x2), a))]


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # put your code here
    a = parse_coefficients(a_1)
    b = parse_coefficients(a_2)

    solutions = common_solutions(a, b)

    return solutions


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    if len(x) == 0:
        raise ValueError("Выборка пуста.")

    data = np.array(x)

    # Считаем среднее
    mean = np.mean(data)

    # Считаем стандартное отклонение
    stddev = np.std(data, ddof=0)

    skewness = np.sum((data - mean) ** 3) / (len(data) * stddev ** 3)

    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here

    data = np.array(x)

    mean = np.mean(data)

    stddev = np.std(data, ddof=0)

    kurtosis = np.sum((data - mean) ** 4) / (len(data) * stddev ** 4) - 3

    return round(kurtosis, 2)
