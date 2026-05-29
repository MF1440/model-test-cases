import itertools
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Starlink720 import Starlink720


def main():
    # Смена текущего каталога для корректного сохранения рисунков (не в корень проекта)
    scriptPath = os.path.dirname(__file__)
    os.chdir(scriptPath)

    # Инициализация генератора случаных чисел для повторяемости результатов
    seed = 8
    np.random.seed(seed)
    random.seed(seed)

    # Создание объекта типа Starlink720, он самостоятенльно подргузить все данные
    # из конфига и выполнит начальную конфигурацию
    group720 = Starlink720()

    # Построим граф связей между КА для выбранного момента времени
    epoch = 583
    group720.rebuildGraph(epoch=epoch)
    group720.isValid()

    # Найдем кратчайший путь между двумя случайными КА
    srcSatIdx, dstSatIdx = np.random.choice(group720.satCount, 2, replace=False)
    path = group720.findShortestPath(srcSatIdx, dstSatIdx)
    print(f"{srcSatIdx} -> {dstSatIdx}: {path}, {len(path)} хопа(ов)")

    # Построим 3D-визуализацию полученного графа нашей орбитальной группы
    group720.plot3d()

    graphVariants = [1, 2, 3, 4, 5, 6]

    # Построим 2D и 3D визуализации графов, построенных разными алгоритмами
    print()
    for currentVariant in graphVariants:
        fileName2d = f"figures/graphVariant{currentVariant:02}-2d.pdf"
        fileName3d = f"figures/graphVariant{currentVariant:02}-3d.png"

        group720.rebuildGraph(epoch=epoch, graphVariant=currentVariant)
        group720.plot2d(outputFileName=fileName2d, verbose=True)
        group720.plot3d(outputFileName=fileName3d, verbose=True)

    # Сравним "качество" работы разных алгоритмов построения графа путем оценки
    # (средней) длины кратчайшего пути
    shortestPathLengthEvaluate(
        group720,
        epoch=epoch,
        graphVariants=[0] + graphVariants,
        outputFileName="figures/pathEvaluateFull.pdf",
    )
    shortestPathLengthEvaluate(
        group720,
        epoch=epoch,
        graphVariants=[0] + graphVariants,
        srcDstPairsCount=2000,
        outputFileName="figures/pathEvaluateSmall.pdf",
    )

    linkLengthEvaluate(
        group720,
        epoch=epoch,
        graphVariants=[0],
        kdeplot=False,
        outputFileName="figures/linkLengthEvaluate0.pdf",
    )

    linkLengthEvaluate(
        group720,
        epoch=epoch,
        graphVariants=graphVariants,
        outputFileName="figures/linkLengthEvaluate.pdf",
    )


def shortestPathLengthEvaluate(
    group720: Starlink720,
    epoch: int,
    graphVariants: List[int] = None,
    srcDstPairsCount: int = -1,
    outputFileName: str = None,
):
    """
    Оценка длины кратчайшего пути для разных типов графа

    Args:
        group720 (Starlink720): объект орбитальной группы

        epoch (int): эпоха (время)

        graphVariants (List[int], optional): список типов графа для оценки. Если \
        не задан (настройка по умолчанию), то будут использованы все доступные.

        srcDstPairsCount (int, optional): число пар вершин графа для поиска \
        кратчайших путей. Если не задано (настройка по умолчанию), то будут \
        использованы все возможные пары.

        outputFileName (str, optional): имя файла для сохранения результата. \
            Если имя не задано (настройка по умолчанию), то данные будет показаны \
            во всплывающем окне.
    """
    satCount = group720.satCount
    fullPairsCount = satCount**2

    pairsCount = int(srcDstPairsCount)
    if (pairsCount < 0) or (pairsCount >= fullPairsCount):
        pairsCount = fullPairsCount
        dstSatList = range(satCount)
        testSet = [dstSatList for _ in range(satCount)]
    else:
        testPairs = list(itertools.product(range(satCount), range(satCount)))
        testPairs = random.sample(testPairs, pairsCount)

        testSet = [[] for _ in range(satCount)]
        for srcSatIdx, dstIdx in testPairs:
            testSet[srcSatIdx].append(dstIdx)

    if not graphVariants:
        graphVariants = [1, 2, 3, 4, 5, 6]

    plt.close()
    fig, ax = plt.subplots(1, 1)

    print()
    print("Оценка длины кратчайшего пути для разных типов графа, ", end="")
    if pairsCount >= fullPairsCount:
        title = f"все возможные пары вершин ({fullPairsCount} шт)"
        print(title + ":")
    else:
        title = f"случайное подмножество пар вершин ({pairsCount}/{fullPairsCount} шт)"
        print(title + ":")

    for currentVariant in graphVariants:
        group720.rebuildGraph(epoch, currentVariant)

        shortestDistances = []
        maxDistance = 1

        for srcSatIdx, dstSatList in enumerate(testSet):
            if len(dstSatList) == 0:
                continue

            fullSearch = len(dstSatList) > 1
            group720.findShortestPath(srcSatIdx, dstSatList[0], fullSearch=fullSearch)

            unreachableCount = 0
            for dstSatIdx in dstSatList:
                distance = group720.shortestDistance[dstSatIdx]
                if distance > 0:
                    maxDistance = max(maxDistance, distance)
                    shortestDistances.append(distance)
                elif distance < 0:
                    unreachableCount += 1

            if unreachableCount:
                print(f"  source {srcSatIdx:2}: ", end="")
                print(f"{unreachableCount:3}/{satCount:3} sats are unreachable")

        shortestDistances = np.array(shortestDistances)

        line = f"  graphVariant = {currentVariant}:"
        line += f" max = {shortestDistances.max():2}"
        line += f" mean = {shortestDistances.mean():5.2f}"
        line += f" std = {shortestDistances.std():.2e}"
        print(line)

        sns.kdeplot(
            shortestDistances,
            gridsize=maxDistance,
            ax=ax,
            label=f"graphVariant = {currentVariant}",
        )

    ax.set_title(title)
    ax.set_xlabel("Расстояние (хопы)")
    ax.legend()

    if outputFileName:
        fig.savefig(outputFileName)
    else:
        plt.show()


def linkLengthEvaluate(
    group720: Starlink720,
    epoch: int,
    graphVariants: List[int] = None,
    outputFileName: str = None,
    kdeplot: bool = True,
):
    """
    Оценка длины линков для разных типов графа

    Args:
        group720 (Starlink720): объект орбитальной группы

        epoch (int): эпоха (время)

        graphVariants (List[int], optional): список типов графа для оценки. Если \
        не задан (настройка по умолчанию), то будут использованы все доступные.

        outputFileName (str, optional): имя файла для сохранения результата. \
            Если имя не задано (настройка по умолчанию), то данные будет показаны \
            во всплывающем окне.
    """

    plt.close()
    fig, ax = plt.subplots(1, 1)

    print()
    print("Оценка длины линков")

    for currentVariant in graphVariants:
        group720.rebuildGraph(epoch, currentVariant)

        srcIdxList = []
        dstIdxList = []
        for srcIdx in range(group720.satCount):
            srcIdxList += [srcIdx] * len(group720.adj[srcIdx])
            dstIdxList += group720.adj[srcIdx]

        srcState = group720.stateEci[srcIdxList, :]
        dstState = group720.stateEci[dstIdxList, :]
        linkLength = np.linalg.norm(srcState - dstState, axis=1) / 1000.0
        print(srcState.shape, linkLength.shape)

        if kdeplot:
            sns.kdeplot(
                linkLength,
                # gridsize=maxDistance,
                ax=ax,
                label=f"graphVariant = {currentVariant}",
            )
        else:
            sns.histplot(
                linkLength,
                # gridsize=maxDistance,
                ax=ax,
                label=f"graphVariant = {currentVariant}",
            )

    ax.set_xlabel("Длина линка (км)")
    ax.legend()

    if outputFileName:
        fig.savefig(outputFileName)
    else:
        plt.show()


if __name__ == "__main__":
    main()
