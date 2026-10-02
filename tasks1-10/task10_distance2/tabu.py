import numpy as np
from graph import Graph
from settings import (REPAIR_MOVES, TABU_TENURE_MIN,
                      TABU_TENURE_MAX, TABU_CONFLICT_FACTOR, PRINT_EVERY)


def choose_move(colors: np.ndarray, counts: np.ndarray,
                conflicts_per_vertex: np.ndarray, tabu_until: np.ndarray,
                iteration: int, conflicts: int, best_conflicts: int,
                rng: np.random.Generator) -> tuple[int, int, int] | None:
    """Выбирает допустимую перекраску с наименьшим изменением конфликтов.
    Проверка всех конфликтующих вершин и всех других цветов.

    :param colors: Текущие цвета вершин.
    :param counts: Массив N x K; counts[v, c] — число соседей v цвета c.
    :param conflicts_per_vertex: Число одноцветных соседей каждой вершины.
    :param tabu_until: Массив N x K итераций окончания запретов.
    :param iteration: Номер текущей перекраски.
    :param conflicts: Текущее число конфликтующих рёбер.
    :param best_conflicts: Лучшее число конфликтов за эту попытку.
    :param rng: Генератор случайных чисел.
    :return: Тройка (вершина, новый цвет, изменение конфликтов).
    """
    best_delta = float("inf")
    moves = []
    for vertex in np.flatnonzero(conflicts_per_vertex):
        vertex = int(vertex)
        old_color = int(colors[vertex])
        for color in range(counts.shape[1]):
            if color == old_color:
                continue

            delta = int(counts[vertex, color] - counts[vertex, old_color])
            forbidden = tabu_until[vertex, color] > iteration
            improves_record = conflicts + delta < best_conflicts
            if forbidden and not improves_record:
                continue

            if delta < best_delta:
                best_delta = delta
                moves = [(vertex, color)]
            elif delta == best_delta:
                moves.append((vertex, color))

    if not moves:
        return None
    vertex, color = moves[int(rng.integers(len(moves)))]
    return vertex, color, int(best_delta)


def tabu_search(graph: Graph, colors: np.ndarray, num_colors: int,
                seed: int) -> tuple[np.ndarray, int]:
    """Исправляет конфликты и сохраняет лучшую найденную раскраску.

    :param graph: Граф запретов G^2.
    :param colors: Массив N цветов 0..K-1.
    :param num_colors: Количество разрешённых цветов K.
    :param seed: Начальное значение генератора случайных чисел.
    :return: Копия лучшей раскраски и число её конфликтующих рёбер.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros((graph.point_count, num_colors), dtype=int)
    conflicts_per_vertex = np.zeros(graph.point_count, dtype=int)

    for vertex in range(graph.point_count):
        neighbors = graph.neighbors(vertex)
        counts[vertex] = np.bincount(colors[neighbors], minlength=num_colors)
        conflicts_per_vertex[vertex] = counts[vertex, colors[vertex]]

    conflicts = int(conflicts_per_vertex.sum() // 2)
    print(f"  После удаления цвета: {conflicts} конфликтов.", flush=True)
    if conflicts == 0:
        print("  Перекраски не потребовались.", flush=True)
        return colors.copy(), 0

    best_conflicts = conflicts
    best_colors = colors.copy()
    tabu_until = np.zeros((graph.point_count, num_colors), dtype=int)
    moves_made = 0

    for iteration in range(REPAIR_MOVES):
        if conflicts == 0:
            break

        move = choose_move(colors, counts, conflicts_per_vertex, tabu_until,
                           iteration, conflicts, best_conflicts, rng)
        if move is None:
            print("  Допустимых перекрасок нет.", flush=True)
            break
        vertex, new_color, delta = move
        old_color = int(colors[vertex])
        colors[vertex] = new_color

        for neighbor in graph.neighbors(vertex):
            counts[neighbor, old_color] -= 1
            counts[neighbor, new_color] += 1
            conflicts_per_vertex[neighbor] = counts[neighbor, colors[neighbor]]
        conflicts_per_vertex[vertex] = counts[vertex, new_color]
        conflicts += delta

        tenure = int(rng.integers(TABU_TENURE_MIN, TABU_TENURE_MAX + 1))
        tenure += TABU_CONFLICT_FACTOR * conflicts
        tabu_until[vertex, old_color] = iteration + tenure

        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_colors = colors.copy()

        moves_made = iteration + 1
        if moves_made % PRINT_EVERY == 0:
            print(f"  Ход {moves_made} из {REPAIR_MOVES}: сейчас {conflicts} конфликтов, "
                  f"лучшее число: {best_conflicts}.", flush=True)

    if best_conflicts > 0 and moves_made == REPAIR_MOVES:
        print("  Достигнут лимит перекрасок.", flush=True)
    print(f"  Выполнено перекрасок: {moves_made}. "
          f"В лучшей раскраске конфликтов: {best_conflicts}.", flush=True)

    return best_colors, best_conflicts
