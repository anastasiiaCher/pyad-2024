import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    import numpy as np

    matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

    def multiply(A, B):
        if len(A[0]) != len(B):
            return "Невозможно перемножить матрицы!"

        result = [[0 for d in range(len(B[0]))] for d in range(len(A))]

        for i in range(len(A)):
            for k in range(len(B[0])):
                for j in range(len(B)):
                    result[i][k] += A[i][j] * B[j][k]
        return result

    result = multiply(matrix1, matrix2)
    print("Произведение матриц:")
    for row in result:
        print(row)
    pass


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    import numpy as np
    from scipy.optimize import minimize_scalar
    def F(x, a11, a12, a13):
        return a11 * x ** 2 + a12 * x + a13

    def P(x, a21, a22, a23):
        return a21 * x ** 2 + a22 * x + a23

    def extremum(a, b, c):
        return -b / (2 * a)

    def find(coeffs1, coeffs2):
        # Получаем коэффициенты первой и второй функции
        a11, a12, a13 = map(float, coeffs1.split())
        a21, a22, a23 = map(float, coeffs2.split())

        # Находим экстремумы функций
        exF = extremum(a11, a12, a13)
        exP = extremum(a21, a22, a23)

        # Проверим, равны ли экстремумы
        if np.isclose(exF, exP, atol=1e-5):
            print(f"У функций одинаковые экстремумы: x = {exF}")
        else:
            print(f"Экстремум F(x) = {exF}, экстремум P(x) = {exP}")

        # Чтобы найти пересечение решений, решаем уравнение: a11 * x^2 + a12 * x + a13 = a21 * x^2 + a22 * x + a23
        A = a11 - a21
        B = a12 - a22
        C = a13 - a23
        discriminant = B ** 2 - 4 * A * C
        if A == 0 and B == 0:
            if C == 0:
                print("Уравнение имеет бесконечно много решений.")
            else:
                print("Уравнение не имеет решений.")
        elif A == 0:
            # Уравнение линейное: B * x + C = 0
            solution = -C / B
            print(f"Уравнение имеет одно решение: x = {solution}")
        else:
            # Уравнение квадратное, смотрим на дискриминант
            if discriminant > 0:
                x1 = (-B + np.sqrt(discriminant)) / (2 * A)
                x2 = (-B - np.sqrt(discriminant)) / (2 * A)
                print(f"Уравнение имеет два решения: x1 = {x1}, x2 = {x2}")
            elif discriminant == 0:
                x = -B / (2 * A)
                print(f"Уравнение имеет одно решение: x = {x}")
            else:
                print("Уравнение не имеет действительных решений.")

    coef1 = input("Введите коэффициенты для функции F(x): ")
    coef2 = input("Введите коэффициенты для функции P(x): ")
    find(coef1, coef2)
    pass


    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    from scipy.stats import kurtosis, skew
    import numpy as np

    def moments(data):
        n = len(data)
        mean = np.mean(data)
        m2 = np.sum((data - mean) ** 2) / n
        m3 = np.sum((data - mean) ** 3) / n
        m4 = np.sum((data - mean) ** 4) / n
        return m2, m3, m4

    def calculate(data):
        m2, m3, m4 = moments(data)
        sigma = np.sqrt(m2)
        skewness = m3 / sigma ** 3
        kurt = m4 / sigma ** 4 - 3
        return skewness, kurt

    # Проверка
    def check(data):
        check_skew = skew(data)
        check_kurt = kurtosis(data)
        return check_skew, check_kurt

    # Пример использования
    np.random.seed(109)
    sample_data = np.random.normal(0, 1, 1000)

    # Рассчитаем коэффициенты асимметрии и эксцесса
    skewness, kurt = calculate(sample_data)
    print(f"Коэффициент асимметрии: {skewness}")
    print(f"Коэффициент эксцесса: {kurt}")

    # Проверим результаты через scipy
    check_skew, check_kurt = check(sample_data)
    print(f"Коэффициент асимметрии (scipy): {check_skew}")
    print(f"Коэффициент эксцесса (scipy): {check_kurt}")


