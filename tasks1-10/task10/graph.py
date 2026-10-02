import math
import numpy as np
from scipy.spatial import cKDTree


class AngularGraph:
    """Граф близких центров.

    :param points: Массив N x 3: координаты центров на единичной сфере.
    :param threshold_deg: Порог T в градусах, 0 < T <= 180.
    :param tree: KD-tree (k-dimensional tree), дерево поиска по этим points.
    """

    def __init__(self, points: np.ndarray, threshold_deg: float, tree: cKDTree) -> None:
        """Подготавливает поиск соседей при фиксированном пороге T.

        :param points: Массив N × 3: координаты центров на единичной сфере.
        :param threshold_deg: Порог T в градусах, 0 < T <= 180.
        :param tree: KD-tree (k-dimensional tree), дерево поиска по этим points.
        :return: None (Сохраняет радиус поиска и степени вершин).
        """
        self.points = points
        self.tree = tree
        self.point_count = len(points)
        self.radius = 2 * math.sin(math.radians(threshold_deg) / 2)
        self.degrees = tree.query_ball_point(points, self.radius, return_length=True) - 1

    def neighbors(self, vertex: int) -> np.ndarray:
        """Находит соседей вершины внутри заданного углового порога.

        :param vertex: Индекс вершины от 0 до N - 1.
        :return: Массив целых индексов соседей без самой вершины.
            Граница порога включена: расстояние по хорде <= радиуса поиска.
        """
        nearby = self.tree.query_ball_point(self.points[vertex], self.radius, return_sorted=False)
        nearby = np.array(nearby)
        return nearby[nearby != vertex]  # Убираем саму вершину.
