import numpy as np
from graph import AngularGraph
from settings import (CANDIDATE_VERTICES, REPAIR_MOVES,
                      TABU_TENURE_MIN, TABU_TENURE_MAX)


def tabu_search(graph: AngularGraph, colors: np.ndarray, conflicts_per_vertex: np.ndarray,
                num_colors: int, seed: int) -> np.ndarray:
    """Исправляет конфликты перекрасками и возвращает лучшее назначение.
    Может временно увеличивать число конфликтов. Остановка: ноль конфликтов,
    лимит REPAIR_MOVES или отсутствие допустимого хода.

    :param graph: Граф близких центров при фиксированном пороге T.
    :param colors: Массив N текущих цветов 0..K-1.
    :param conflicts_per_vertex: Массив N чисел одноцветных соседей.
        Изменяется на месте и относится к текущему, а не лучшему состоянию.
    :param num_colors: Количество доступных цветов K.
    :param seed: Начальное число генератора случайных чисел.
    :return: Копия цветов с наименьшим числом конфликтов за эту попытку.
    """
    rng = np.random.default_rng(seed)
    conflicts = int(conflicts_per_vertex.sum() // 2)
    best_conflicts = conflicts
    best_colors = colors.copy()
    tabu = {}  # (вершина, цвет) -> итерация, когда снова разрешён этот цвет.

    for iteration in range(REPAIR_MOVES):
        if conflicts == 0:
            break
        move, delta = choose_move(graph, colors, conflicts_per_vertex,
                                  conflicts, best_conflicts, tabu,
                                  iteration, num_colors, rng)
        if move is None:
            break

        vertex, old_color, new_color, neighbors = move
        old_matches = neighbors[colors[neighbors] == old_color]
        new_matches = neighbors[colors[neighbors] == new_color]
        conflicts_per_vertex[old_matches] -= 1
        conflicts_per_vertex[new_matches] += 1
        conflicts_per_vertex[vertex] = len(new_matches)
        colors[vertex] = new_color
        conflicts += delta

        tenure = int(rng.integers(TABU_TENURE_MIN, TABU_TENURE_MAX + 1))
        tabu[(vertex, old_color)] = iteration + tenure
        for assignment in list(tabu):
            if tabu[assignment] <= iteration:
                del tabu[assignment]

        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_colors = colors.copy()

    print("Табу-поиск: осталось конфликтов", best_conflicts, flush=True)
    return best_colors


def choose_move(graph: AngularGraph, colors: np.ndarray, conflicts_per_vertex: np.ndarray,
                conflicts: int, best_conflicts: int, tabu: dict[tuple[int, int], int],
                iteration: int, num_colors: int, rng: np.random.Generator
                ) -> tuple[tuple[int, int, int, np.ndarray] | None, int | float]:
    """Выбирает допустимую перекраску с наименьшим изменением конфликтов.
    Рассматривает несколько конфликтных вершин и все цвета для каждой.

    :param graph: Граф при текущем пороге T.
    :param colors: Массив N текущих цветов 0..K-1.
    :param conflicts_per_vertex: Массив N чисел конфликтов у отдельных вершин.
    :param conflicts: Общее число конфликтных рёбер в текущем состоянии.
    :param best_conflicts: Наименьшее число конфликтов за эту попытку.
    :param tabu: (вершина, цвет) -> номер итерации окончания запрета.
    :param iteration: Номер текущего хода, начиная с 0.
    :param num_colors: Количество доступных цветов K.
    :param rng: Генератор для выбора вершин и разрешения равенств.
    :return: Пара (move, delta). move — (вершина, старый цвет, новый цвет,
        соседи); delta — изменение числа конфликтов, которое может быть >= 0.
        Если допустимого хода нет, возвращает (None, inf).
    """
    conflicting = np.flatnonzero(conflicts_per_vertex)
    sample_size = min(CANDIDATE_VERTICES, len(conflicting))
    candidates = rng.choice(conflicting, size=sample_size, replace=False)
    worst_vertex = int(np.argmax(conflicts_per_vertex))
    candidates = np.unique(np.append(candidates, worst_vertex))

    best_move = None
    best_delta = float("inf")
    for vertex in candidates:
        vertex = int(vertex)
        neighbors = graph.neighbors(vertex)
        old_color = int(colors[vertex])
        counts = np.bincount(colors[neighbors], minlength=num_colors)
        deltas = counts - counts[old_color]
        allowed = np.ones(num_colors, dtype=bool)
        allowed[old_color] = False

        for (tabu_vertex, tabu_color), expiry in tabu.items():
            if tabu_vertex == vertex and expiry > iteration:
                # Запрет нарушается только ради лучшего варианта.
                if conflicts + deltas[tabu_color] >= best_conflicts:
                    allowed[tabu_color] = False

        valid = np.flatnonzero(allowed)
        if len(valid) == 0:
            continue
        delta = int(deltas[valid].min())
        tied = valid[deltas[valid] == delta]
        new_color = int(rng.choice(tied))

        if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
            best_delta = delta
            best_move = (vertex, old_color, new_color, neighbors)

    return best_move, best_delta
