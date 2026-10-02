import math
import numpy as np


def fibonacci_sphere(count: int) -> np.ndarray:
    """Распределяет точки по единичной сфере по алгоритму Фибоначчи.

    :param count: Количество точек N, положительное целое число.
    :return: Массив N × 3 единичных векторов с координатами x, y, z.
    """
    indices = np.arange(count)
    z = 1 - 2 * (indices + 0.5) / count
    golden_angle = math.pi * (3 - math.sqrt(5))
    longitude = np.remainder(indices * golden_angle, 2 * math.pi)
    radius_section = np.sqrt(np.maximum(0, 1 - z * z))
    x = radius_section * np.cos(longitude)
    y = radius_section * np.sin(longitude)
    points = np.column_stack((x, y, z))
    return points / np.linalg.norm(points, axis=1, keepdims=True)

def angular_distance_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Вычисляет углы между соответствующими строками двух массивов.

    :param first: Массив M x 3 ненулевых векторов.
    :param second: Массив M x 3 ненулевых векторов.
    :return: Массив M углов в градусах, от 0 до 180.
    """
    sine = np.linalg.norm(np.cross(first, second), axis=1)
    cosine = np.sum(first * second, axis=1)
    return np.degrees(np.arctan2(sine, cosine))

# def angular_distance_deg(first, second) -> np.floating | np.ndarray:
#     """Вычисляет угол между векторами из центра сферы.

#     :param first: Первый ненулевой вектор (3,) или массив векторов M × 3.
#     :param second: Второй ненулевой вектор или массив соответствующих векторов.
#     :return: Угол в градусах для одной пары либо массив углов для нескольких пар.
#     """
#     first = np.asarray(first)
#     second = np.asarray(second)
#     sine = np.linalg.norm(np.cross(first, second), axis=-1)
#     cosine = np.sum(first * second, axis=-1)
#     return np.degrees(np.arctan2(sine, cosine))

def minimum_angle_deg(points: np.ndarray, colors: np.ndarray, num_colors: int) -> float:
    """Проверяет все пары одного цвета и вычисляет наименьший угол.

    :param points: Массив N x 3: координаты центров на единичной сфере.
    :param colors: Массив N цветов от 0 до num_colors - 1.
    :param num_colors: Количество цветов K.
    :return: Минимальный угол в градусах.
    """
    minimum = float("inf")
    for color in range(num_colors):
        group = points[colors == color]
        if len(group) < 2:
            continue
        # Создаем индексы разных пар: (0, 1), (0, 2), ..., (1, 2), ...
        first, second = np.triu_indices(len(group), k=1)
        a = group[first]
        b = group[second]
        angles = angular_distance_deg(a, b)
        minimum = min(minimum, float(angles.min()))
    return minimum

def upper_bound_deg(point_count: int, color_count: int) -> float:
    """Вычисляет верхнюю оценку Фейеша Тота для минимального угла.
    Хотя бы один цвет встречается ceil(N / K) раз. Оценка для этой группы
    ограничивает возможный результат всей раскраски.

    :param point_count: Количество центров N.
    :param color_count: Количество цветов K.
    :return: Верхняя оценка в градусах, для чисел задания около 27.320757°.
    """
    group_size = math.ceil(point_count / color_count)
    if group_size == 2:
        return 180.0
    omega = math.pi * group_size / (6 * group_size - 12)
    cotangent = 1 / math.tan(omega)
    return math.degrees(math.acos((cotangent * cotangent - 1) / 2))
