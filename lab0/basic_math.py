# Задание №1
def matrix_multiplication(matrix_a, matrix_b):
    # Проверяем возможность умножения: число столбцов matrix1 должно быть равно числу строк matrix2
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы не могут быть перемножены: размерности не соответствуют.")

    result = [[0] * len(matrix_b[0]) for _ in range(len(matrix_a))]

    # Проход по строкам matrix1
    for i in range(len(matrix_a)):
        # Проход по столбцам matrix2
        for k in range(len(matrix_b[0])):
            # Вычисление каждого элемента новой матрицы
            for j in range(len(matrix_b)):
                result[i][k] += matrix_a[i][j] * matrix_b[j][k]

    return result


# Задание №2

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # put your code here
    pass


# Задание №3
def skew(x):
    if len(x) == 0:
        return None  # Возвращаем None для пустой выборки
    mean = sum(x) / len(x)
    m2 = sum((num - mean) ** 2 for num in x) / len(x)  # Момент второго порядка
    m3 = sum((num - mean) ** 3 for num in x) / len(x)  # Момент третьего порядка
    skewness = m3 / (m2 ** (3 / 2)) if m2 != 0 else 0  # Проверка на ноль в знаменателе
    return round(skewness, 2)  # Округляем до 2 знаков после запятой


def kurtosis(x):
    if len(x) == 0:
        return None  # Возвращаем None для пустой выборки
    mean = sum(x) / len(x)
    m2 = sum((num - mean) ** 2 for num in x) / len(x)  # Момент второго порядка
    m4 = sum((num - mean) ** 4 for num in x) / len(x)  # Момент четвертого порядка
    kurt = m4 / (m2 ** 2) - 3 if m2 != 0 else 0  # Проверка на ноль в знаменателе
    return round(kurt, 2)  # Округляем до 2 знаков после запятой