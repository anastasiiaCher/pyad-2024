from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import minimize_scalar


def matrix_multiplication(
    matrix_a: List[List[float]], matrix_b: List[List[float]]
) -> List[List[float]]:
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError()

    # Транспонируем чтобы не извращаться с индексами вложенных списков
    # Будем просто перемножать строки УДОБНО!
    matrix_b = list(zip(*matrix_b))

    result = []

    for row_a in matrix_a:
        row = []
        for row_b in matrix_b:
            row.append(sum(a * b for a, b in zip(row_a, row_b)))
        result.append(row)

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    first, second = list(map(float, a_1.split())), list(map(float, a_2.split()))
    # Грубо говоря, просто идем в обратном порядке, так как на последним коэфом идет свободный член
    first_func = lambda x: sum(
        [coff * pow(x, index) for index, coff in enumerate(reversed(first))]
    )
    second_func = lambda x: sum(
        [coff * pow(x, index) for index, coff in enumerate(reversed(second))]
    )

    # Выплюнет Nan если нет экстремума и предупреждение в stderr из-за большого кол-ва
    # суммирования
    extremum_first = minimize_scalar(
        first_func,
    )
    extremum_second = minimize_scalar(
        second_func,
    )
    print("Экстремумы первой: %s, %s" % (extremum_first.x, extremum_second.fun))
    print("Экстремум второй: %s, %s" % (extremum_second.x, extremum_second.fun))

    intersected_args = [a - b for a, b in zip(first, second)]

    if all(x == 0 for x in intersected_args):
        return None

    roots = np.roots(intersected_args)
    # sad that i need to sort this :(
    return sorted([(root, first_func(root)) for root in roots], key=lambda x: x[0])


def hardcoded_functions(a_1, a_2):
    """
    Тут решение которое захардкожено только под квадратные уравнения
    Обладает похожей идеей XD
    """

    @dataclass
    class PolynomialArgs:
        a: float
        b: float
        c: float

    first_func = lambda x: first_args.a * x * x + first_args.b * x + first_args.c
    second_func = lambda x: second_args.a * x * x + second_args.b * x + second_args.c

    # put your code here
    first_func_args = list(map(float, a_1.split()))
    second_func_args = list(map(float, a_2.split()))

    first_args = PolynomialArgs(
        first_func_args[0], first_func_args[1], first_func_args[2]
    )
    second_args = PolynomialArgs(
        second_func_args[0], second_func_args[1], second_func_args[2]
    )

    # extremum_first = minimize_scalar(
    #     first_func,
    # )
    # extremum_second = minimize_scalar(
    #     second_func,
    # )
    # print("Экстремумы первой: %s" % extremum_first)
    # print("Экстремум второй: %s" % extremum_second)
    # так как по определению задачи мы работаем ТОЛЬКО с квадратичными уравнениями
    # для нахождения общих решений просто воспользуемся уранвнеием
    # a_1x^2 + b_1x + c_1 = a_2x^2 + b_2x + c_1
    intersection = PolynomialArgs(
        a=first_args.a - second_args.a,
        b=first_args.b - second_args.b,
        c=first_args.c - second_args.c,
    )

    # Полностью совпадают
    if not intersection.a and not intersection.b and not intersection.c:
        return None
    # Различаются только сдвигом по Y
    if intersection.a == 0 and intersection.b == 0:
        return []

    # Случаи когда уравнение приходит в линейное
    if not intersection.b:
        root = -intersection.c / intersection.a
        return [(root, second_func(root))]

    if not intersection.a:
        root = -intersection.c / intersection.b
        return [(root, first_func(root))]

    # Решение обычного квадратичного уравнения
    discriminant = intersection.b**2 - 4 * intersection.a * intersection.c
    if discriminant < 0:
        return []

    if discriminant == 0:
        root = -intersection.b / (2 * intersection.a)
        return [root, first_func(root)]

    first_root = (-intersection.b - discriminant**0.5) / (2 * intersection.a)
    second_root = (-intersection.b + discriminant**0.5) / (2 * intersection.a)
    return sorted(
        [
            (first_root, first_func(first_root)),
            (second_root, second_func(second_root)),
        ],
        key=lambda d: d[0],
    )


def find_moment(samples: List[float], moment: int):
    average = sum(samples) / len(samples)
    return sum([(sample - average) ** moment for sample in samples]) / len(samples)


def skew(x: List[float]):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    third_moment = find_moment(x, 3)
    dispersion = find_moment(x, 2)
    result = third_moment / (dispersion ** (3 / 2))
    return round(result, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    fourth_moment = find_moment(x, 4)
    dispersion = find_moment(x, 2)
    result = fourth_moment / (dispersion ** (4 / 2)) - 3
    return round(result, 2)
