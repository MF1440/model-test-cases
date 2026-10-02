import numpy as np
from graph import Graph


def dsatur(graph: Graph) -> np.ndarray:
    """Раскрашивает граф методом DSATUR — Degree of Saturation, насыщение.
    Выбирает вершину по насыщению, числу нераскрашенных соседей и индексу.
    Назначает первый свободный цвет; при необходимости открывает новый.

    :param graph: Граф запретов G^2.
    :return: Массив N цветов.
    """
    count = graph.point_count
    capacity = int(graph.degrees.max()) + 1
    colors = np.full(count, -1, dtype=int)
    forbidden = np.zeros((count, capacity), dtype=bool)
    priorities = graph.degrees.copy()

    for _ in range(count):
        vertex = int(np.argmax(priorities))
        neighbors = graph.neighbors(vertex)
        color = int(np.flatnonzero(~forbidden[vertex])[0])

        uncolored = neighbors[colors[neighbors] < 0]
        new_color_neighbors = uncolored[~forbidden[uncolored, color]]
        forbidden[uncolored, color] = True
        priorities[uncolored] -= 1
        priorities[new_color_neighbors] += count

        colors[vertex] = color
        priorities[vertex] = -1
    return colors


def remove_color(graph: Graph, colors: np.ndarray, seed: int) -> np.ndarray:
    """Удаляет цвет с наименьшим числом вершин
    и распределяет его вершины по оставшимся цветам.

    :param graph: Граф запретов G^2.
    :param colors: Полная раскраска цветами 0..K-1.
    :param seed: Начальное значение генератора случайных чисел.
    :return: Копия раскраски с цветами 0..K-2, возможно с конфликтами.
    """
    rng = np.random.default_rng(seed)
    result = colors.copy()
    sizes = np.bincount(result)
    removed = int(np.argmin(sizes))
    vertices = rng.permutation(np.flatnonzero(result == removed))

    result[vertices] = -1
    result[result > removed] -= 1
    for vertex in vertices:
        neighbor_colors = result[graph.neighbors(vertex)]
        assigned = neighbor_colors[neighbor_colors >= 0]
        counts = np.bincount(assigned, minlength=len(sizes) - 1)
        candidates = np.flatnonzero(counts == counts.min())
        result[vertex] = int(rng.choice(candidates))
    return result
