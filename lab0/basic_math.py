from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """

    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Matrices cannot be multiplied: wrong size.")

    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    def poly(abc, arg):
        return abc[0] * arg ** 2 + abc[1] * arg + abc[2]

    def res(arg):
        return arg, poly(a_1, arg)

    def extremum(coefficients):
        minimum = minimize_scalar(lambda f: poly(coefficients, f))
        return minimum.x, poly(coefficients, minimum.x)

    a_1 = list(map(int, a_1.split()))
    a_2 = list(map(int, a_2.split()))

    extremum_1 = extremum(a_1)
    extremum_2 = extremum(a_2)

    a, b, c = [a_1[i] - a_2[i] for i in range(3)]
    d = b * b - 4 * a * c

    if a == 0:
        if b == 0:
            if c == 0:
                return None
            return []
        x = -c / b
        return [(res(x))]

    if d == 0:
        x = -b / a / 2
        return [(res(x))]
    elif d > 0:
        d = d ** 0.5
        x1, x2 = (-b + d) / a / 2, (-b - d) / a / 2
        return [(res(x1)), (res(x2))]
    return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    return skew_kurtosis(x)[0]


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    return skew_kurtosis(x)[1]


def skew_kurtosis(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    deviance = variance ** 0.5

    skew_value = sum((v - mean) ** 3 for v in values) / n / (deviance ** 3)
    kurtosis_value = sum((v - mean) ** 4 for v in values) / n / (deviance ** 4) - 3

    return round(skew_value, 2), round(kurtosis_value, 2)
