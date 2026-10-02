import numpy as np
from graph import AngularGraph


def dsatur(graph: AngularGraph, num_colors: int) -> tuple[np.ndarray, np.ndarray]:
    """Раскрашивает граф методом DSATUR.
    Выбирает вершину по насыщению, числу нераскрашенных соседей и индексу.

    :param graph: Граф близких центров при одном фиксированном пороге T.
    :param num_colors: Количество доступных цветов K.
    :return: Пара массивов (colors, conflicts_per_vertex) длины N.
        colors содержит цвета 0..K-1; второй массив — число одноцветных соседей.
    """
    count = graph.point_count
    colors = np.full(count, -1)
    forbidden = np.zeros((count, num_colors), dtype=bool)
    sizes = np.zeros(num_colors)
    conflicts_per_vertex = np.zeros(count)
    priorities = graph.degrees.astype(np.int64).copy()

    for _ in range(count):
        vertex = int(np.argmax(priorities))
        neighbors = graph.neighbors(vertex)
        available = np.flatnonzero(~forbidden[vertex])

        if len(available) > 0:
            color = int(available[np.argmin(sizes[available])])
        else:
            neighbor_colors = colors[neighbors]
            assigned = neighbor_colors[neighbor_colors >= 0]
            counts = np.bincount(assigned, minlength=num_colors)
            candidates = np.flatnonzero(counts == counts.min())
            color = int(candidates[np.argmin(sizes[candidates])])

        matching = neighbors[colors[neighbors] == color]
        conflicts_per_vertex[matching] += 1
        conflicts_per_vertex[vertex] = len(matching)

        uncolored = neighbors[colors[neighbors] < 0]
        new_color_neighbors = uncolored[~forbidden[uncolored, color]]
        forbidden[uncolored, color] = True
        priorities[uncolored] -= 1
        priorities[new_color_neighbors] += count

        colors[vertex] = color
        sizes[color] += 1
        priorities[vertex] = -1

    print("DSATUR: конфликтов", int(conflicts_per_vertex.sum() // 2), flush=True)
    return colors, conflicts_per_vertex