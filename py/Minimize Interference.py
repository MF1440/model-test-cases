import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import SphericalVoronoi
from constellation import *

def findEdgesFromSphericalVoronoi(points, radius, center):
    sv = SphericalVoronoi(points, radius, center)
    sv.sort_vertices_of_regions()

    edgeToGenerators = {}
    for pIdx, region in enumerate(sv.regions):
        m = len(region)
        if m < 2:
            continue
        for k in range(m):
            a = region[k]
            b = region[(k + 1) % m]
            edge = frozenset((a, b))
            edgeToGenerators.setdefault(edge, []).append(pIdx)

    edges = set()
    for gens in edgeToGenerators.values():
        if len(gens) == 2:
            i, j = gens
            if i != j:
                edges.add(tuple(sorted((i, j))))

    return sorted(edges)

def buildVoronoiGraph(points, constellation, radius=1, center=np.array([0, 0, 0])):
    edges = findEdgesFromSphericalVoronoi(points, radius, center)
    voronoiGraph = nx.Graph()
    
    satToGroup = []

    cumulativeSatCount = 0
    for groupIdx, group in enumerate(constellation.groups):
        cumulativeSatCount += group.getTotalSatCount()
        satToGroup.append((cumulativeSatCount, groupIdx))

    initialFreq = {}
    prevSatIdxTh = 0
    satToGroupIter = iter(satToGroup)
    curSatToGroup = next(satToGroupIter)
    for i, point in enumerate(points):
        satIdxTh, groupIdx = curSatToGroup
        if i >= satIdxTh:
            prevSatIdxTh = satIdxTh
            curSatToGroup = next(satToGroupIter)
            satIdxTh, groupIdx = curSatToGroup        
        group = constellation.groups[groupIdx]
        satIdxInGroup = i - prevSatIdxTh
        satIdxIsOdd = satIdxInGroup % 2
        planeIdxIsOdd = (satIdxInGroup // group.satsPerPlane) % 2
        # Назначаем 2 чередующиеся частоты {0,1} всем КА на четных орбитальных плоскостях,
        # а 2 другие чередующиеся частоты {2,3} - всем КА на нечетных орбитальных плоскостях.
        # Здесь возможны конфликты, если кол-во КА в орбитальной плоскости нечетно - их разрешит основной алгоритм.
        # (хотя можно добавить еще 2 частоты {5,6} для таких случаев)
        initialFreq[tuple(point)] = satIdxIsOdd + 2 * planeIdxIsOdd
    
    for i, v in enumerate(points):
        voronoiGraph.add_node(i, coords=v, freq=initialFreq[tuple(v)])

    voronoiGraph.add_edges_from(edges)

    return voronoiGraph

def colorizeGreedy(graph, maxColors=4):
    ordering = []
    degrees = dict(graph.degree())
    graphCopy = graph.copy()

    while graphCopy.nodes:
        v = min(graphCopy.nodes, key=lambda x: degrees[x])
        ordering.append(v)
        for neighbor in graphCopy.neighbors(v):
            degrees[neighbor] -= 1
        graphCopy.remove_node(v)

    coloring = {}
    for v in reversed(ordering):
        usedColors = {coloring.get(n) for n in graph.neighbors(v) if n in coloring}
        coloring[v] = min(c for c in range(maxColors) if c not in usedColors)

    return coloring

def minimizeFreqSwitches(graphs, numFrequencies):
    tCount = len(graphs)
    freqAssignments = []
    prevAssignment = {node: data['freq'] for node, data in graphs[0].nodes.items()}
    totalSwitches = 0

    for t in range(tCount):
        graph = graphs[t]
        assignment = {}

        for node, data in graph.nodes.items():
            neighbors = list(graph.neighbors(node))
            neighborFreqs = {prevAssignment.get(n) for n in neighbors}
            neighborFreqs.discard(None)

            prevFreq = prevAssignment.get(node)
            if prevFreq is not None and prevFreq not in neighborFreqs:
                assignment[node] = prevFreq
            else:
                bestFreq = None
                minConflict = float('inf')
                minSwitchPenalty = float('inf')

                for f in range(numFrequencies):
                    conflict = sum(1 for n in neighbors if prevAssignment.get(n) == f)
                    switchPenalty = 0 if prevFreq == f else 1

                    if conflict < minConflict or (conflict == minConflict and switchPenalty < minSwitchPenalty):
                        bestFreq = f
                        minConflict = conflict
                        minSwitchPenalty = switchPenalty

                assignment[node] = bestFreq

        if t > 0:
            switches = sum(1 for n in assignment if assignment[n] != prevAssignment.get(n))
            totalSwitches += (switches)

        freqAssignments.append(assignment)
        prevAssignment = assignment

    return freqAssignments, totalSwitches

def plotPointsOnSphere(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='gray')

    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, color='red', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Точки на сфере')
    ax.set_box_aspect([1,1,1])

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Решение задачи минимизации интерференции и переключений частот")
    parser.add_argument("-c", "--constellation", default="singlePolar", help="Название группировки КА (по-умолчанию - singlePolar)")
    parser.add_argument("-f", "--freqs", type=int, default=10, help="Кол-во доступных частот (по-умолчанию - 10)")
    parser.add_argument("-t", "--epochs", type=int, default=1002, help="Кол-во эпох")
    parser.add_argument("-p", "--plot", action="store_true", help="Вывод трехмерного изображения точек на единичной сфере в нулевую эпоху")
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # создание объекта типа Constellation, инициализация параметрами группировки Stalink из конфига
    constellation = Constellation(args.constellation)

    # вычисление элементов орбиты для всех КА в начальный момент
    constellation.getInitialState()

    # определение точек на оси времени, в которые будут проихзводиться расчёты
    epochs = list(range(args.epochs))

    # расчёт положений всех КА в заданные моменты времени
    constellation.propagateJ2(epochs)

    if args.plot:
        coverageRegionCenters = constellation.stateEci[:, :, 0]
        # coverageRegionCenters = np.transpose(coverageRegionCenters)
        coverageRegionCenters /= np.linalg.norm(coverageRegionCenters, axis=1)[:, np.newaxis]
        plotPointsOnSphere(coverageRegionCenters)
    else:
        numFreqs = args.freqs
        voronoiGraphs = []
        for epochIdx in range(len(epochs)):
            coverageRegionCenters = constellation.stateEci[:, :, epochIdx]
            coverageRegionCenters /= np.linalg.norm(coverageRegionCenters, axis=1)[:, np.newaxis]
            
            print(f"\rСтроим граф Вороного в момент времени t = {epochIdx:4d}... {epochIdx / len(epochs):3.0%}", end="")
            voronoiGraphs.append(buildVoronoiGraph(coverageRegionCenters, constellation))
        print()

        freqAssignments, freqSwitchCount = minimizeFreqSwitches(voronoiGraphs, numFreqs)

        print(f"Частоту пришлось переключить в {freqSwitchCount / (constellation.totalSatCount * len(epochs)):3.2%} случаев")


if __name__ == "__main__":
    main()
