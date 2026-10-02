import json
from pathlib import Path
import numpy as np

def save_result(points: np.ndarray, color_of_points: np.ndarray, report: dict) -> None:
    """Сохраняет центры, номера и отчёт в папку result.

    :param points: Массив N × 3: координаты центров на единичной сфере.
    :param color_of_points: Массив N цветов от 1 до K (colors + 1).
    :param report: Словарь с минимальным углом и сведениями о поиске.
    :return: None. Записывает cells.npz и report.json, заменяя старые файлы.
    """
    directory = Path("result")
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / "cells.npz", unit_centers=points, color_of_points=color_of_points)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (directory / "report.json").write_text(text + "\n", encoding="utf-8")
