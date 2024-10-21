from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError

    rows_res_len = len(matrix_a)
    cols_res_len = len(matrix_b[0])
    res = []

    for i in range(rows_res_len):
        res.append([])
        for j in range(cols_res_len):
            sm = 0
            for k in range(len(matrix_b)):
                sm += (matrix_a[i][k] * matrix_b[k][j])
            res[i].append(sm)
    return res

def functions(a_1, a_2):
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    def f(x, a, b, c):
        return a * x ** 2 + b * x + c

    def find_extremum(a, b, c):
        def g(x):
            return a * x ** 2 + b * x + c

        result = minimize_scalar(g)
        return result.x, f(result.x, a, b, c)

    a1, b1, c1 = map(int, a_1.split())
    a2, b2, c2 = map(int, a_2.split())

    extremum1 = find_extremum(a1, b1, c1)
    extremum2 = find_extremum(a2, b2, c2)

    print(extremum1, extremum2)

    a = a1 - a2
    b = b1 - b2
    c = c1 - c2

    if a == 0 and b == 0 and c == 0:
        return None

    if a != 0:
        d = b ** 2 - 4 * a * c
        if d < 0:
            return []
        else:
            x1 = (-b + d ** 0.5) / 2 * a
            x2 = (-b - d ** 0.5) / 2 * a
            print(x1, x2, a, b, c)
            if x1 == x2:
                return [x1, a1 * x1 ** 2 + b1 * x1 + c1]
            else:
                return [(x1, a1 * x1 ** 2 + b1 * x1 + c1), (x2, a1 * x2 ** 2 + b1 * x2 + c1)]
    elif b != 0:
        x = -c / b
        return [(x, f(x, a, b, c))]
    else:
        return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    x_i = sum(x) / len(x)
    sigma = (sum([(num - x_i) ** 2 for num in x]) / len(x)) ** 0.5
    m3 = (sum([(num - x_i) ** 3 for num in x]) / len(x))
    a3 = m3 / sigma ** 3
    return round(a3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x_i = sum(x) / len(x)
    sigma = (sum([(num - x_i) ** 2 for num in x]) / len(x)) ** 0.5
    m4 = (sum([(num - x_i) ** 4 for num in x]) / len(x))
    e4 = m4 / sigma ** 4 - 3
    return round(e4, 2)
