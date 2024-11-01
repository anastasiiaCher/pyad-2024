import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        print("Умножение выполнить невозможно")
    else:
        result = [[0 for i in range(len(matrix_b[0]))] for i in range(len(matrix_a))]

        for i in range(len(matrix_a)):  
            for j in range(len(matrix_b[0])):
                for k in range(len(matrix_b)):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]

        for row in result:
            print(row)


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # Записывает коэффициенты из строки
    a11, a12, a13 = map(float, coeffs_F)
    a21, a22, a23 = map(float, coeffs_P)

    # Определим функции
    def F(x):
        return a11 * x**2 + a12 * x + a13

    def P(x):
        return a21 * x**2 + a22 * x + a23

    # Поиск экстремума через -b / (2a):
    def extremum(a, b):
        return -b / (2 * a)

    # Находим экстремумы
    if a11 != 0:
        extremum_F = extremum(a11, a12)
    else:
        None
    if a21 != 0:
        extremum_P = extremum(a21, a22)
    else:
        None

    # Выводим экстремумы
    if a11 != 0:
        print(f"Точка экстремума F(x): {extremum_F}")
    else:
        print("Функция F(x) линейная") 
    if a21 != 0:
        print(f"Точка экстремума P(x): {extremum_P}")
    else:
        print("Функция P(x) линейная") 

    # Решаем уравнение F(x) = P(x) и находим дискриминант
    A = a11 - a21
    B = a12 - a22
    C = a13 - a23

    discriminant = B**2 - 4 * A * C

    # Анализируем дискриминант для определения количества решений
    if A == 0 and B == 0 and C == 0:
        print("Решений бесконечно много (F(x) и P(x) совпадают).")
    elif A == 0 and B == 0:
        print("Решений нет (F(x) и P(x) — параллельные (если так можно сказать про параболы)).")
    elif A == 0:
        x = -C / B
        print(f"Есть одно решение: x = {x}")
    elif discriminant > 0:
        x1 = (-B + np.sqrt(discriminant)) / (2 * A)
        x2 = (-B - np.sqrt(discriminant)) / (2 * A)
        print(f"Есть два решения: x1 = {x1}, x2 = {x2}")
    elif discriminant == 0:
        x = -B / (2 * A)
        print(f"Есть одно решение: x = {x}")
    else:
        print("Решений нет.")


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(sample)  # Объем выборки
    mean = np.mean(sample)  # Выборочное среднее

    # Центральные моменты
    m2 = np.sum((sample - mean)**2) / n  # Момент второго порядка (дисперсия)
    m3 = np.sum((sample - mean)**3) / n  # Момент третьего порядка

    # Стандартное отклонение (корень из m2)
    sigma = np.sqrt(m2)

    # Коэффициент асимметрии A3
    A3 = m3 / sigma**3

    # Вывод результатов
    print(f"Коэффициент асимметрии (A3): {A3}")


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(sample)  # Объем выборки
    mean = np.mean(sample)  # Выборочное среднее

    # Центральные моменты
    m2 = np.sum((sample - mean)**2) / n  # Момент второго порядка (дисперсия
    m4 = np.sum((sample - mean)**4) / n  # Момент четвертого порядка

    # Стандартное отклонение (корень из m2)
    sigma = np.sqrt(m2)

    # Коэффициент эксцесса E4
    E4 = m4 / sigma**4 - 3

    # Вывод результатов
    print(f"Коэффициент эксцесса (E4): {E4}")
