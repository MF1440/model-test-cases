import numpy as np
from scipy.spatial import ConvexHull


class Graph:
    """Граф с массивом соседей для каждой вершины.

    :param adjacency: Список N массивов индексов соседей.
    """

    def __init__(self, adjacency: list[np.ndarray]) -> None:
        self.adjacency = adjacency
        self.point_count = len(adjacency)
        self.degrees = np.array([len(row) for row in adjacency], dtype=int)

    def neighbors(self, vertex: int) -> np.ndarray:
        """Находит соседей выбранной вершины.

        :param vertex: Индекс вершины от 0 до N - 1.
        :return: Возвращает массив индексов соседей без самой вершины.
        """
        return self.adjacency[vertex]


def cell_graph(points: np.ndarray) -> Graph:
    """Соединяет центры сот, имеющих общую сторону.
    По построееным точкам алгортмом Фибоначчи строим выпуклую оболочку,
    ребра которой показывают соседство сферических сот Вороного.

    :param points: Массив N x 3 центров на единичной сфере.
    :return: Граф непосредственного соседства G.
    """
    hull = ConvexHull(points) # Выпуклая оболочка 
    adjacency = [set() for _ in points]
    for first, second, third in hull.simplices:
        adjacency[first].update((int(second), int(third)))
        adjacency[second].update((int(first), int(third)))
        adjacency[third].update((int(first), int(second)))
    return Graph([np.array(sorted(row), dtype=int) for row in adjacency])


def square_graph(graph: Graph) -> Graph:
    """Соединяет вершины на расстоянии двух переходов.

    :param graph: Граф непосредственного соседства G.
    :return: Граф запретов G^2.
    """
    adjacency = []
    for vertex in range(graph.point_count):
        nearby = set(graph.neighbors(vertex))
        for neighbor in graph.neighbors(vertex):
            nearby.update(graph.neighbors(neighbor))
        nearby.discard(vertex)
        adjacency.append(np.array(sorted(nearby), dtype=int))
    return Graph(adjacency)


def conflict_degrees(graph: Graph, colors: np.ndarray) -> np.ndarray:
    """Считает одноцветных соседей каждой вершины.

    :param graph: Граф запретов G^2.
    :param colors: Массив N назначенных цветов от 0.
    :return: Массив N счётчиков.
    """
    degrees = np.zeros(graph.point_count, dtype=int)
    for vertex in range(graph.point_count):
        neighbors = graph.neighbors(vertex)
        degrees[vertex] = np.count_nonzero(colors[neighbors] == colors[vertex])
    return degrees


def check_coloring(graph: Graph, colors: np.ndarray) -> int:
    """Проверяет разные цвета у каждой соты и всех её соседей.
    Любые две соты на расстоянии не более двух входят хотя бы
    в одно такое окружение.

    :param graph: Граф G.
    :param colors: Массив N назначенных цветов.
    :return: Число окружений с повтором цвета.
    """
    violations = 0
    for vertex in range(graph.point_count):
        neighborhood = [vertex, *graph.neighbors(vertex)]
        if len(np.unique(colors[neighborhood])) != len(neighborhood):
            violations += 1
    return violations
