import os
import sys
from collections import deque
from typing import List, Optional

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

_scriptPath = os.path.dirname(__file__)
sys.path.append(os.path.realpath(f"{_scriptPath}/../py"))

from Constellation import Constellation, Earth  # noqa: E402


class Starlink720:
    """
    Класс, реализующий построение графа связей между КА орбитальной группы Starlink
    для эшелона 720 км. В текущей реализации есть участки, существенно завязанные
    на характеристики граппы, например число КА в каждой плоскости. Учёт такой
    "конкретной" информации необходим, например, при непосредственном построении
    графа разными методами `self._rebuildV`.

    Другой пример использования специфики конкретной орбитальной группы - это
    конфигурация её орбитальных плоскостей. При просмотре на эти плоскости со
    стороны экватора наблюдается отчетливая X-образная "конструкция" из 4 пар
    плоскостей: 0-4, 1-5, 2-6 и 3-7. Для упрощения написания кода адресации КА
    на этих X-образных пар были реализованы вспомогательные методы `self._northLeftIdx()`,
    `self._northRightIdx()`,`self._southLeftIdx()` и `self._southRightIdx()`,
    каждый из которых отвечает за соответствующую "часть" пары.
    """

    def __init__(self):
        self._constellation = Constellation("Starlink")
        self._constellation.calcInitialState()

        group720Idx = 1
        self._group = self._constellation.groups[group720Idx]
        assert self._group.altitude == 720
        assert self._group.satsPerPlane == 50

        self._stateEci = np.zeros(0)
        """Координаты КА в инерциальной системе координат ECI (Earth-centered inertial)"""

        self._stateStartIdx = 0
        for currentGroup in self._constellation.groups[:group720Idx]:
            self._stateStartIdx += currentGroup.totalSatCount()
        self._stateEndIdx = self._stateStartIdx + self._group.totalSatCount()

        satCount = self._group.totalSatCount()
        planeCount = self._group.planeCount

        self.A = np.zeros([satCount, satCount], dtype=np.bool)
        """Матрица смежности"""

        self.adj: List[List[int]] = [[] for _ in range(satCount)]
        """Список смежности"""

        self.shortestPath: List[int] = []
        """Последний найденный кратчайший путь"""

        self.shortestDistance = np.full(satCount, -1, dtype=int)
        """Длина кратчайшего пути (число хопов) для каждого узла графа"""

        self._linksCount = np.zeros(satCount)
        self._ascendingNodeIdx = np.full(planeCount, -1, dtype=int)
        """Индекс спутника, наиболее близкого к восходящему узлу"""

        self._rebuildFunc = None

        # вспомогательные переменные, используются для (2d) отрисовки графа
        self._ax: Optional[matplotlib.axes.Axes] = None
        self._axStack = []
        self._axPlaneStep = 5
        self._axPlaneShift = 1
        self._axMaxUsedPlane = 1
        self._axLinkKw = {
            "marker": "o",
            "markersize": 4,
            "markeredgecolor": "black",
            "markerfacecolor": "white",
        }

    @property
    def satCount(self) -> int:
        """Общее число спутников группы"""
        return self._group.totalSatCount()

    @property
    def planeCount(self) -> int:
        """Число орбитальных плоскостей группы"""
        return self._group.planeCount

    @property
    def satsPerPlane(self) -> int:
        """Число КА в каждой орбитальной плоскости группы"""
        return self._group.satsPerPlane

    def rebuildGraph(self, epoch: int, graphVariant: int = 6):
        """
        Выполняет построение/перестроение графа для пространственнного положения
        КА в выбранный момент времени (эпоху). Результаты работы сохраняются в
        полях self.A (матрица смежности) и self.adj (список смежности).

        Args:
            epoch (int): эпоха (время)

            graphVariant (int, optional): Вариант алгоритма перестроения графа, \
            значение из диапазона [1, 6]. По умолчанию выбран алгоритм номер 6

        Raises:
            ValueError: в случае выбора недопустимого значения параметра graphVariant
        """

        assert epoch >= 0

        constellation = self._constellation
        stateStartIdx = self._stateStartIdx
        stateEndIdx = self._stateEndIdx

        satCount = self.satCount
        satsPerPlane = self.satsPerPlane
        planeCount = self.planeCount

        constellation.propagateJ2([epoch])
        self._stateEci = constellation.stateEci[stateStartIdx:stateEndIdx, :, 0]

        self.A.fill(False)
        self.adj = [[] for _ in range(satCount)]
        self.shortestPath = []
        self.shortestDistance.fill(-1)

        self._linksCount.fill(0)
        self._ascendingNodeIdx.fill(-1)

        satFlatIndex = self._satFlatIndex
        addLink = self._addLink

        for planeIdx in range(planeCount):
            for satInPlaneIdx in range(satsPerPlane):
                sat1Idx = satFlatIndex(planeIdx, satInPlaneIdx)
                sat2Idx = satFlatIndex(planeIdx, satInPlaneIdx + 1)
                addLink(sat1Idx, sat2Idx)

                sat1z = self._stateEci[sat1Idx, 2]
                sat2z = self._stateEci[sat2Idx, 2]
                if (sat1z * sat2z <= 0) and (sat1z < sat2z):
                    if abs(sat1z) < abs(sat2z):
                        nearestIdx = satInPlaneIdx
                    else:
                        nearestIdx = (satInPlaneIdx + 1) % satsPerPlane
                    self._ascendingNodeIdx[planeIdx] = nearestIdx

        match graphVariant:
            case 0:
                self._rebuildFunc = None
                return

            case 1:
                self._rebuildFunc = self._rebuildV1

            case 2:
                self._rebuildFunc = self._rebuildV2

            case 3:
                self._rebuildFunc = self._rebuildV3

            case 4:
                self._rebuildFunc = self._rebuildV4

            case 5:
                self._rebuildFunc = self._rebuildV5

            case 6:
                self._rebuildFunc = self._rebuildV6

            case _:
                raise ValueError()

        for planeIdx in range(planeCount):
            self._rebuildFunc(planeIdx)

    def findShortestPath(
        self,
        srcSatIdx: int,
        dstSatIdx: int,
        fullSearch: bool = False,
    ) -> List[int]:
        """
        Выполняет поиск кратчайшего пути от одного спутника до другого. Используется
        алгоритм поиска в ширину (BFS), т.к. наш граф - невзвешенный. Сложность
        в нашем случае равна O(V + E) = O(V + 3V) = O(V).

        Результаты работы сохраняются в поля self.shortestPath (кратчайший путь)
        и self.shortestDistance (расстояния от старта до всех просмотренных вершин
        графа).

        Args:
            srcSatIdx (int): начальная точка пути (номер спутника)
            dstSatIdx (int): начальная точка пути (номер спутника)
            fullSearch (bool, optional): выполнять ли полный обход доступных вершин
            графа (если выбрано True), либо остановиться по достижению заданной
            конечной точки (если выбрано False). По умолчанию полный поиск отключен
            (задано False).

        Returns:
            List[int]: найденный кратчайший путь, либо пустой список (если
            такого пути не существует)
        """

        satCount = self.satCount
        assert (srcSatIdx >= 0) and (srcSatIdx < satCount)
        assert (dstSatIdx >= 0) and (dstSatIdx < satCount)

        nodeQueue = deque()
        parent = np.full(satCount, -1, dtype=int)

        shortestDistance = self.shortestDistance
        shortestDistance.fill(-1)

        nodeQueue.append(srcSatIdx)
        shortestDistance[srcSatIdx] = 0
        self.shortestPath = []

        while nodeQueue:
            currentSatIdx = nodeQueue.popleft()
            currentDistance = shortestDistance[currentSatIdx]

            if (not fullSearch) and (currentSatIdx == dstSatIdx):
                break

            for nextSatIdx in self.adj[currentSatIdx]:
                if shortestDistance[nextSatIdx] < 0:
                    parent[nextSatIdx] = currentSatIdx
                    shortestDistance[nextSatIdx] = currentDistance + 1
                    nodeQueue.append(nextSatIdx)

        shortestPath = deque()
        currentSatIdx = dstSatIdx
        while currentSatIdx >= 0:
            shortestPath.appendleft(int(currentSatIdx))
            currentSatIdx = parent[currentSatIdx]

        if shortestPath[0] == srcSatIdx:
            self.shortestPath = list(shortestPath)

        return self.shortestPath

    def plot2d(self, outputFileName: str = None, verbose: bool = False):
        """
        Выполняет 2d отрисовку текущего варианта графа. Полученный рисунок содержит
        только рёбра (связи) для орбитальной плоскости 0, для остальных плоскостей
        результат будет идентичным.

        Args:
            outputFileName (str, optional): имя файла для сохранения результата. \
            Если имя не задано (настройка по умолчанию), то данные будет показаны \
            во всплывающем окне.

            verbose (bool, optional): включение/отключение печати сообщений в \
            терминал. По умолчанию выключено (False).
        """

        plt.close()
        fig, ax = plt.subplots(1, 1)

        ax.hlines([0, 12.5], -10, 100, "black", ls=":")

        planeStep = self._axPlaneStep
        planeShift = self._axPlaneShift

        yMin = -1
        yMax = 13
        for rightPlaneIdx in range(4):
            for idx, (planeIdx, color) in enumerate(
                [
                    (rightPlaneIdx + 4, "tab:green"),
                    (rightPlaneIdx, "tab:blue"),
                ]
            ):
                planePos = rightPlaneIdx * planeStep + planeShift * idx

                ax.vlines(planePos, yMin, yMax, color=color)
                ax.text(planePos, 13.4, f"{planeIdx}", ha="center")

                ax.plot(
                    [planePos] * (yMax - yMin - 1),
                    range(yMin + 1, yMax),
                    color=color,
                    **self._axLinkKw,
                )

        # ----------------------------------------------------------------------

        self._ax = ax
        self._axStack = []
        self._axMaxUsedPlane = 1

        if self._rebuildFunc:
            self._rebuildFunc(0)

        ax.grid(alpha=0.3)
        ax.set_ylim(top=(yMax + 1.5), bottom=(yMin - 1.5))

        leftLimit = -0.3 * self._axPlaneStep
        rightLimit = (self._axMaxUsedPlane + 0.5) * self._axPlaneStep
        ax.set_xlim(left=leftLimit, right=rightLimit)

        ax.set_xticks([])

        if outputFileName:
            fig.savefig(outputFileName)
            if verbose:
                print(outputFileName)
        else:
            fig.show()

        # ----------------------------------------------------------------------

        self._ax = None
        self._axStack = []
        self._axMaxUsedPlane = 1

    def plot3d(
        self,
        satColor: str = "black",
        linkColor: str = "black",
        pathColor: str = "red",
        earthColor: str = "light_blue",
        satSize: float = 1e5,
    ):
        """
        Выполняет 3d отрисовку текущей орбитальной группы, включая планету,
        спутники, связи между ними (рёбра графа), а также последний найденный
        кратчайший путь. Положение спутников соответствует последнему запуску
        метода `self.rebuildGraph()`

        Значения для параметров, задающих цвета, могут быть найдены по ссылке:
        https://docs.pyvista.org/api/utilities/named_colors.html

        Args:
            satColor (str, optional): цвет спутников. Значение по умолчанию: "black".

            linkColor (str, optional): цвет рёбер графа. Значение по умолчанию: "black".

            pathColor (str, optional): цвет последнего найденного кратчайшего пути. Значение по умолчанию: "red".

            earthColor (str, optional): цвет планеты Земля. Значение по умолчанию: "light_blue".

            satSize (float, optional): размер спутников. Значение по умолчанию: 1e5.
        """

        sats = []
        links = []
        for srcIdx in range(self.satCount):
            sats.append(
                pv.Cube(
                    center=self._stateEci[srcIdx, :],
                    x_length=satSize,
                    y_length=satSize,
                    z_length=satSize,
                )
            )

            for dstIdx in range(self.satCount):
                if self.A[srcIdx, dstIdx] != 0:
                    links.append(
                        pv.Line(
                            self._stateEci[srcIdx, :],
                            self._stateEci[dstIdx, :],
                        )
                    )

        shortestPath = []
        if self.shortestPath:
            shortestPathTail = self.shortestPath
            while len(shortestPathTail) > 1:
                sat1Idx = shortestPathTail[0]
                sat2Idx = shortestPathTail[1]
                shortestPath.append(
                    pv.Line(
                        self._stateEci[sat1Idx, :],
                        self._stateEci[sat2Idx, :],
                    )
                )
                shortestPathTail = shortestPathTail[1:]

        pl = pv.Plotter()

        pl.add_mesh(pv.Sphere(radius=Earth.radius), color=earthColor)

        if links:
            pl.add_mesh(pv.merge(links), color=linkColor)

        if sats:
            pl.add_mesh(pv.merge(sats), color=satColor)

        if shortestPath:
            pl.add_mesh(
                pv.merge(shortestPath),
                color=pathColor,
                line_width=2,
            )

        pl.show_axes()

        cameraPos = pv.CameraPosition(
            position=(Earth.radius * 6, 0.0, 0.0),
            focal_point=(0.0, 0.0, 0.0),
            viewup=(0.0, 0.0, 1.0),
        )
        pl.show(cpos=cameraPos)

    def isValid(self, verbose: bool = True) -> bool:
        """
        Проверяет корректность построенного графа. Может быть использовано при,
        например разработке новых вариантов алгоритмов построения графа по текущим
        координатам КА.

        Args:
            verbose (bool, optional): включение/отключение печати сообщений в \
            терминал с указанием найденной проблемы. По умолчанию включено (True).

        Returns:
            bool: True если граф корректен, False в ином случае.
        """

        if self._linksCount.max() > 3:
            if verbose:
                print("Граф содержит узлы с числом связей, превышающим 3")

            return False

        for srcSatIdx in range(self.satCount):
            for dstSatIdx in self.adj[srcSatIdx]:
                middlePoint = 0.5 * (
                    self._stateEci[srcSatIdx] + self._stateEci[dstSatIdx]
                )

                if np.linalg.norm(middlePoint) <= Earth.radius:
                    if verbose:
                        print("Граф содержит рёбра, пересекающие поверхность Земли")

                    return False

        return True

    def _rebuildV1(self, planeIdx: int):
        addLink = self._addLink
        northLeftIdx = self._northLeftIdx
        northRightIdx = self._northRightIdx
        southLeftIdx = self._southLeftIdx
        southRightIdx = self._southRightIdx

        # длинный горизонтальный линк по экватору
        addLink(
            northRightIdx(planeIdx, 0),
            northLeftIdx(planeIdx + 1, 0),
        )

        for idxFunc1, idxFunc2 in [
            (northRightIdx, northLeftIdx),
            (southRightIdx, southLeftIdx),
        ]:
            # короткий линк между ближайшими плоскостями
            for satIdx in [1, 3, 5, 7, 9, 11]:
                addLink(
                    idxFunc1(planeIdx, satIdx),
                    idxFunc2(planeIdx, satIdx),
                )

            # длинный горизонтальный линк к соседней (+1) плоскости
            for satIdx in [2, 4, 6, 8, 10, 12]:
                addLink(
                    idxFunc1(planeIdx, satIdx),
                    idxFunc2(planeIdx + 1, satIdx),
                )

    def _rebuildV2(self, planeIdx: int):
        addLink = self._addLink
        northLeftIdx = self._northLeftIdx
        northRightIdx = self._northRightIdx
        southLeftIdx = self._southLeftIdx
        southRightIdx = self._southRightIdx

        # длинный горизонтальный линк по экватору
        addLink(
            northRightIdx(planeIdx, 0),
            northLeftIdx(planeIdx + 1, 0),
        )

        for idxFunc1, idxFunc2 in [
            (northRightIdx, northLeftIdx),
            (southRightIdx, southLeftIdx),
        ]:
            # короткий линк между ближайшими плоскостями
            for satIdx in [2, 5, 8, 11]:
                addLink(
                    idxFunc1(planeIdx, satIdx),
                    idxFunc2(planeIdx, satIdx),
                )

            # длинные диагональные линки к соседней (+1) плоскости
            for satIdx in [1, 4, 7, 10]:
                addLink(
                    idxFunc1(planeIdx, satIdx),
                    idxFunc2(planeIdx + 1, satIdx + 2),
                )

                addLink(
                    idxFunc1(planeIdx, satIdx + 2),
                    idxFunc2(planeIdx + 1, satIdx),
                )

    def _rebuildV3(self, planeIdx: int):
        addLink = self._addLink
        northLeftIdx = self._northLeftIdx
        northRightIdx = self._northRightIdx
        southLeftIdx = self._southLeftIdx
        southRightIdx = self._southRightIdx

        # длинный горизонтальный линк по экватору
        addLink(
            northRightIdx(planeIdx, 0),
            northLeftIdx(planeIdx + 1, 0),
        )

        for idxFunc1, idxFunc2 in [
            (northRightIdx, northLeftIdx),
            (southRightIdx, southLeftIdx),
        ]:
            # короткий линк между ближайшими плоскостями
            for satIdx in [1, 3, 5, 7, 9, 11]:
                addLink(
                    idxFunc1(planeIdx, satIdx),
                    idxFunc2(planeIdx, satIdx),
                )

            # длинные диагональные линки к соседней (+1) плоскости
            for satIdx in [2, 6, 10]:
                addLink(
                    idxFunc1(planeIdx, satIdx),
                    idxFunc2(planeIdx + 1, satIdx + 2),
                )

                addLink(
                    idxFunc1(planeIdx, satIdx + 2),
                    idxFunc2(planeIdx + 1, satIdx),
                )

    def _rebuildV4(self, planeIdx: int):
        addLink = self._addLink
        northLeftIdx = self._northLeftIdx
        northRightIdx = self._northRightIdx
        southLeftIdx = self._southLeftIdx
        southRightIdx = self._southRightIdx

        # короткий линк по экватору между ближайшими плоскостями
        addLink(
            northLeftIdx(planeIdx, 0),
            northRightIdx(planeIdx, 0),
        )

        # диагональные линки между ближайшими плоскостями
        for idxFunc1, idxFunc2 in [
            (northLeftIdx, northRightIdx),
            (southLeftIdx, southRightIdx),
        ]:
            addLink(
                idxFunc1(planeIdx, 5),
                idxFunc2(planeIdx, 10),
            )
            addLink(
                idxFunc2(planeIdx, 5),
                idxFunc1(planeIdx, 10),
            )

            addLink(
                idxFunc1(planeIdx, -2),
                idxFunc2(planeIdx, 2),
            )
            addLink(
                idxFunc2(planeIdx, -2),
                idxFunc1(planeIdx, 2),
            )

        for idxFunc1, idxFunc2 in [
            (northRightIdx, northLeftIdx),
            (southRightIdx, southLeftIdx),
        ]:
            # длинный горизонтальный линк к соседней (+1) плоскости
            addLink(
                idxFunc1(planeIdx, 1),
                idxFunc2(planeIdx + 1, 1),
            )

        for idxFunc in [
            northLeftIdx,
            northRightIdx,
            southLeftIdx,
            southRightIdx,
        ]:
            # длинные диагональные линки к соседней (+1) плоскости
            addLink(
                idxFunc(planeIdx, 3),
                idxFunc(planeIdx + 1, 4),
            )
            addLink(
                idxFunc(planeIdx, 7),
                idxFunc(planeIdx + 1, 6),
            )

            # длинный линк около полюса к дальней (+2) плоскости
            addLink(
                idxFunc(planeIdx, 9),
                idxFunc(planeIdx + 2, 8),
            )

            # длинный линк около полюса к дальней (+3) плоскости
            addLink(
                idxFunc(planeIdx, 11),
                idxFunc(planeIdx + 3, 12),
            )

    def _rebuildV5(self, planeIdx: int):
        addLink = self._addLink
        northLeftIdx = self._northLeftIdx
        northRightIdx = self._northRightIdx
        southLeftIdx = self._southLeftIdx
        southRightIdx = self._southRightIdx

        # короткий линк по экватору между ближайшими плоскостями
        addLink(
            northLeftIdx(planeIdx, 0),
            northRightIdx(planeIdx, 0),
        )

        # диагональные линки между ближайшими плоскостями
        for satIdx in [1, 7]:
            for idxFunc1, idxFunc2 in [
                (northLeftIdx, northRightIdx),
                (southLeftIdx, southRightIdx),
            ]:
                addLink(
                    idxFunc1(planeIdx, satIdx),
                    idxFunc2(planeIdx, satIdx + 5),
                )
                addLink(
                    idxFunc2(planeIdx, satIdx),
                    idxFunc1(planeIdx, satIdx + 5),
                )

        for idxFunc in [
            northLeftIdx,
            northRightIdx,
            southLeftIdx,
            southRightIdx,
        ]:
            # длинные диагональные линки к соседней (+1) плоскости
            addLink(
                idxFunc(planeIdx, 3),
                idxFunc(planeIdx + 1, 5),
            )
            addLink(
                idxFunc(planeIdx, 4),
                idxFunc(planeIdx + 1, 2),
            )
            addLink(
                idxFunc(planeIdx, 8),
                idxFunc(planeIdx + 1, 8 + 1),
            )

            # длинный линк около полюса к дальней (+3) плоскости
            addLink(
                idxFunc(planeIdx, 11),
                idxFunc(planeIdx + 3, 11 - 1),
            )

    def _rebuildV6(self, planeIdx: int):
        addLink = self._addLink
        northLeftIdx = self._northLeftIdx
        northRightIdx = self._northRightIdx
        southLeftIdx = self._southLeftIdx
        southRightIdx = self._southRightIdx

        # длинный горизонтальный линк по экватору
        addLink(
            northRightIdx(planeIdx, 0),
            northLeftIdx(planeIdx + 1, 0),
        )

        # диагональные линки через экватор, между ближайшими плоскостями
        addLink(
            northLeftIdx(planeIdx, 3 - 6),
            northRightIdx(planeIdx, 3),
        )
        addLink(
            northLeftIdx(planeIdx, 3),
            northRightIdx(planeIdx, 3 - 6),
        )

        for idxFunc1, idxFunc2 in [
            (northLeftIdx, northRightIdx),
            (southLeftIdx, southRightIdx),
        ]:
            if planeIdx < (self.planeCount // 2):
                # диагональные линки через полюс, между ближайшими плоскостями
                addLink(
                    idxFunc1(planeIdx, 10 + 5),
                    idxFunc2(planeIdx, 10),
                )
                addLink(
                    idxFunc2(planeIdx, 10 + 5),
                    idxFunc1(planeIdx, 10),
                )

            # диагональные линки между ближайшими плоскостями
            addLink(
                idxFunc1(planeIdx, 4),
                idxFunc2(planeIdx, 4 + 5),
            )
            addLink(
                idxFunc2(planeIdx, 4),
                idxFunc1(planeIdx, 4 + 5),
            )

        for idxFunc in [
            northLeftIdx,
            northRightIdx,
            southLeftIdx,
            southRightIdx,
        ]:
            # длинные диагональные линки к соседней (+1) плоскости
            addLink(
                idxFunc(planeIdx, 1),
                idxFunc(planeIdx + 1, -2),
            )
            addLink(
                idxFunc(planeIdx, 6),
                idxFunc(planeIdx + 1, 6 + 2),
            )
            addLink(
                idxFunc(planeIdx, 7),
                idxFunc(planeIdx + 1, 7 - 2),
            )

            # длинный линк около полюса к дальней (+3) плоскости
            addLink(
                idxFunc(planeIdx, 11),
                idxFunc(planeIdx + 3, 12),
            )

    def _addLink(self, srcIdx: int, dstIdx: int):
        if self._ax:
            x1, y1 = self._axStack.pop()
            x2, y2 = self._axStack.pop()

            # не отображаем данные с южного полушария
            if y1 >= 0 or y2 >= 0:
                self._ax.plot([x1, x2], [y1, y2], "black", **self._axLinkKw)

        if not self.A[srcIdx, dstIdx]:
            self.A[srcIdx, dstIdx] = 1
            self.A[dstIdx, srcIdx] = 1

            self.adj[srcIdx].append(dstIdx)
            self.adj[dstIdx].append(srcIdx)

            self._linksCount[srcIdx] += 1
            self._linksCount[dstIdx] += 1

    def _satFlatIndex(self, planeIdx: int, satInPlaneIdx: int) -> int:
        planeCount = self.planeCount
        satsPerPlane = self.satsPerPlane

        planeIdx = planeIdx % planeCount
        satInPlaneIdx = satInPlaneIdx % satsPerPlane

        return int(planeIdx * satsPerPlane + satInPlaneIdx)

    def _northLeftIdx(self, planeIdx: int, satIdxFromEquator: int) -> int:
        if self._ax:
            axPoint = (
                planeIdx * self._axPlaneStep + 0,
                satIdxFromEquator,
            )
            self._axStack.append(axPoint)
            self._axMaxUsedPlane = max(self._axMaxUsedPlane, planeIdx)

        planeCount = self.planeCount
        halfSatsPerPlane = self.satsPerPlane // 2

        planeIdx = (planeCount // 2 + planeIdx) % planeCount
        satIdx = self._ascendingNodeIdx[planeIdx] + halfSatsPerPlane - satIdxFromEquator

        return self._satFlatIndex(planeIdx, satIdx)

    def _northRightIdx(self, planeIdx: int, satIdxFromEquator: int) -> int:
        if self._ax:
            axPoint = (
                planeIdx * self._axPlaneStep + self._axPlaneShift,
                satIdxFromEquator,
            )
            self._axStack.append(axPoint)
            self._axMaxUsedPlane = max(self._axMaxUsedPlane, planeIdx)

        planeIdx = planeIdx % self.planeCount
        satIdx = self._ascendingNodeIdx[planeIdx] + satIdxFromEquator

        return self._satFlatIndex(planeIdx, satIdx)

    def _southLeftIdx(self, planeIdx: int, satIdxFromEquator: int) -> int:
        return self._northRightIdx(planeIdx, -satIdxFromEquator)

    def _southRightIdx(self, planeIdx: int, satIdxFromEquator: int) -> int:
        return self._northLeftIdx(planeIdx, -satIdxFromEquator)
