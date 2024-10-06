import numpy as np
from scipy.optimize import fsolve


def quadratic_function(coeffs, x):
    a, b, c = coeffs
    return a * x ** 2 + b * x + c


def find_intersections(
        coeffs1: list[float],
        coeffs2: list[float],
        range_: tuple = (-100, 100),
):
    def difference(x):
        return quadratic_function(coeffs1, x) - quadratic_function(coeffs2, x)

    solutions = fsolve(difference, [range_[0], 0, range_[1]])
    intersections = [x for x in solutions if np.isclose(difference(x), 0, atol=1e-5)]

    # Бесконечно много решений
    if np.allclose(coeffs1, coeffs2):
        return None

    intersections = [round(x, 5) for x in intersections]
    return np.unique(intersections) if intersections else []
