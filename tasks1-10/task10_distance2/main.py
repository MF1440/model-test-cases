import json
from pathlib import Path
import numpy as np

from coloring import dsatur, remove_color
from geometry import fibonacci_sphere
from graph import cell_graph, square_graph, check_coloring, Graph
from settings import POINT_COUNT, TARGET_LABELS, ATTEMPTS, SEED
from tabu import tabu_search


def save_result(points: np.ndarray, colors: np.ndarray, direct: Graph,
                history: list[dict]) -> None:
    """Проверка раскраски и сохранение центрово, литер и отчёт.

    :param points: Массив N x 3 центров на единичной сфере.
    :param colors: Массив N цветов от 0; все номера идут подряд.
    :param direct: Граф непосредственного соседства сот.
    :param history: Итоги попыток уменьшить количество цветов.
    :return: None. Записывает cells.npz и report.json в папку result.
    """
    violations = check_coloring(direct, colors)
    if violations != 0:
        raise ValueError("Нельзя сохранить раскраску с конфликтами.")

    report = {
        "point_count": len(points),
        "num_labels": int(colors.max()) + 1,
        "violating_neighborhoods": violations,
        "attempts": history,
    }
    directory = Path("result")
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / "cells.npz",
                        unit_centers=points, labels=colors + 1)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (directory / "report.json").write_text(text + "\n", encoding="utf-8")


def main():
    print(f"Создаём центры сот: {POINT_COUNT} точек.", flush=True)
    points = fibonacci_sphere(POINT_COUNT)
    print("Строим соседство сот и граф запретов на две дистанции.", flush=True)
    direct = cell_graph(points)
    graph = square_graph(direct)

    colors = dsatur(graph)
    num_colors = int(colors.max()) + 1
    history = []
    print(f"DSATUR: правильная начальная раскраска в {num_colors} литер.", flush=True)

    while num_colors > TARGET_LABELS:
        target = num_colors - 1
        improved = False
        print(f"\nПереход с {num_colors} на {target} литер.", flush=True)

        for attempt in range(ATTEMPTS):
            seed = SEED + attempt
            print(f"Попытка {attempt + 1} из {ATTEMPTS}, seed = {seed}.", flush=True)

            candidate = remove_color(graph, colors, seed)
            candidate, conflicts = tabu_search(graph, candidate, target, seed)

            history.append({"num_labels": target,
                            "attempt": attempt + 1, "conflicts": conflicts})
            if conflicts == 0:
                _, colors = np.unique(candidate, return_inverse=True)
                num_colors = int(colors.max()) + 1
                print(f"Получена правильная раскраска в {num_colors} литер.", flush=True)
                improved = True
                break
            print(f"Попытка не удалась: осталось {conflicts} конфликтов.", flush=True)

        if not improved:
            print(f"За {ATTEMPTS} попыток перейти на {target} литер не удалось.", flush=True)
            break

    save_result(points, colors, direct, history)
    print(f"\nИтог: {num_colors} литер. Проверка двух дистанций пройдена.")
    print("Сохранено: result/cells.npz и result/report.json")

main()
