import numpy as np
from scipy.spatial import cKDTree

from coloring import dsatur
from geometry import fibonacci_sphere, minimum_angle_deg, upper_bound_deg
from graph import AngularGraph
from results import save_result
from settings import (ATTEMPTS, COLOR_COUNT, POINT_COUNT, SEED,
                      START_DEG, STEP_DEG)
from tabu import tabu_search


def main():
    """Создаёт центры, выполняет поиск и сохраняет лучший результат.

    Использует константы settings.py. Качество сравнивается по измеренному
    минимальному углу, а не по заданному порогу T.

    :return: None. Записывает result/cells.npz и result/report.json.
    """
    points = fibonacci_sphere(POINT_COUNT)
    tree = cKDTree(points)
    upper_bound = upper_bound_deg(POINT_COUNT, COLOR_COUNT)

    # Просто раздаем всем вершинам по цвету циклично.
    best_colors = np.arange(POINT_COUNT, dtype=np.int32) % COLOR_COUNT
    best_angle = minimum_angle_deg(points, best_colors, COLOR_COUNT)
    threshold = min(START_DEG, upper_bound)
    step = STEP_DEG
    history = []

    for attempt in range(ATTEMPTS):
        print(f"Попытка {attempt + 1}: T = {threshold:.6f}°", flush=True)
        graph = AngularGraph(points, threshold, tree)
        colors, conflicts_per_vertex = dsatur(graph, COLOR_COUNT)
        colors = tabu_search(graph, colors, conflicts_per_vertex,
                             COLOR_COUNT, SEED + attempt)
        angle = minimum_angle_deg(points, colors, COLOR_COUNT)

        if angle > best_angle:
            best_angle = angle
            best_colors = colors.copy()
        print(f"Получено: {angle:.9f}°; лучшее: {best_angle:.9f}°", flush=True)
        history.append({"threshold_deg": threshold, "minimum_angle_deg": angle})

        if best_angle >= upper_bound - 1e-9:
            break
        if angle < threshold - 1e-9:
            step /= 2
        threshold = min(upper_bound, best_angle + step)

    sizes = np.bincount(best_colors)
    report = {
        "point_count": POINT_COUNT,
        "num_labels": COLOR_COUNT,
        "minimum_angle_deg": best_angle,
        "proven_upper_bound_deg": upper_bound,
        "smallest_class": int(sizes.min()),
        "largest_class": int(sizes.max()),
        "attempts": history,
    }
    save_result(points, best_colors + 1, report)
    print("Готово: result/cells.npz и result/report.json")

main()
