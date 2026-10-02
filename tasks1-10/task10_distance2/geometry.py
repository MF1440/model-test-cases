import math
import numpy as np


def fibonacci_sphere(count: int) -> np.ndarray:
    """Распределяем точки по единичной сфере по алгоритму Фибоначчи.

    :param count: Количество точек N, положительное целое число.
    :return: Массив N x 3 единичных векторов с координатами x, y, z.
    """
    indices = np.arange(count)
    z = 1 - 2 * (indices + 0.5) / count
    golden_angle = math.pi * (3 - math.sqrt(5))
    longitude = np.remainder(indices * golden_angle, 2 * math.pi)
    radius = np.sqrt(np.maximum(0, 1 - z * z))
    x = radius * np.cos(longitude)
    y = radius * np.sin(longitude)
    points = np.column_stack((x, y, z))
    return points / np.linalg.norm(points, axis=1, keepdims=True)
