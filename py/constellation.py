import numpy as np
import json
from typing import NamedTuple
from collections import deque

class Parameters(object):
    pass


Const = Parameters()
Const.earthRadius = 6378135      # Экваториальный радиус Земли [m]
Const.earthGM = 3.986004415e+14  # Гравитационный параметр Земли [m3/s2]
Const.earthJ2 = 1.082626e-3      # Вторая зональная гармоника геопотенциала

group = Parameters()


class Walker(NamedTuple):
    inclination: float           # наклонение орбиты
    satsPerPlane: int            # число КА в каждой орбитальной плоскости группы
    planeCount: int              # число орбитальных плоскостей в группе
    f: int                       # фазовый сдвиг по аргументу широты между КА в соседних плоскостях
    altitude: float              # высота орбиты
    maxRaan: float               # максимум прямого восхождения восходящего узла (при распределении орбитальных плоскостей)
    startRaan: float             # прямое восхождение восходящего узла для первой плоскости


class WalkerGroup(Walker):

    def getTotalSatCount(self):
        return self.satsPerPlane * self.planeCount

    def getInitialElements(self):
        startRaan   = np.deg2rad(self.startRaan)
        maxRaan     = np.deg2rad(self.maxRaan)
        inclination = np.deg2rad(self.inclination)
        altitude    = self.altitude * 1000
        satCount    = self.getTotalSatCount()

        raans = np.linspace(startRaan, startRaan + maxRaan, self.planeCount + 1)
        raans = raans[:-1] % (2 * np.pi)

        elements = np.zeros((satCount, 6))
        idx = 0

        for raanIdx, raan in enumerate(raans):
            for satIdx in range(self.satsPerPlane):
                sma = Const.earthRadius + altitude
                aol = 2 * np.pi * (satIdx / self.satsPerPlane + self.f * raanIdx / satCount)

                elements[idx, :] = [sma, 0, 0, raan, inclination, aol]
                idx += 1

        return elements

    def buildAdjacencyMatrixInnerPlaneLinksOnly(self):
        # Link of satellite to itself is considered as 0 in adjacency matrix
        simpleAdjacencyMatrix = np.zeros((self.getTotalSatCount(), self.getTotalSatCount()))
        for planeIdx in range(self.planeCount):
            for satelliteIdx in range(self.satsPerPlane):
                blockIdx = planeIdx * self.satsPerPlane
                idx = blockIdx + satelliteIdx
                shiftIdxFrwd = (satelliteIdx + 1) % self.satsPerPlane
                shiftIdxBkwd = (satelliteIdx - 1 + self.satsPerPlane) % self.satsPerPlane
                simpleAdjacencyMatrix[idx, blockIdx + shiftIdxFrwd] = 1
                simpleAdjacencyMatrix[idx, blockIdx + shiftIdxBkwd] = 1
        return simpleAdjacencyMatrix
        # Matrix is symmetric as 2 linked satellites can receive and send signals to each other.
        # It means that we can store only half of matrix (for example, in upper triangular matrix form).
        # Also matrix is sparse, so we can use CSR (or CSC) format to reduce memory consumption.

    def buildAdjacencyMatrixNeighbourPlanesLinksIncluded(self):
        # Having fixed self.f = 1 (as in example) consider simple neighbour planes links that doesn't take into account
        # that distance between satellites is changing and other effects that can be a problem for linking
        # ( for example, satellites could be on distance more than Earth radius )
        simpleAdjacencyMatrix = np.zeros((self.getTotalSatCount(), self.getTotalSatCount()))
        for planeIdx in range(self.planeCount):
            for satelliteIdx in range(self.satsPerPlane):
                blockIdx = planeIdx * self.satsPerPlane
                idx = blockIdx + satelliteIdx
                # In plane
                shiftIdxFrwd = (satelliteIdx + 1) % self.satsPerPlane
                shiftIdxBkwd = (satelliteIdx - 1 + self.satsPerPlane) % self.satsPerPlane
                # Neighbour planes
                shiftPlaneIdxRight = (planeIdx + 1) % self.planeCount
                shiftPlaneIdxLeft = (planeIdx - 1 + self.planeCount) % self.planeCount
                shiftSatelliteIdxRight = (round(satelliteIdx - self.f / self.planeCount) + self.satsPerPlane) % self.satsPerPlane
                shiftSatelliteIdxLeft = (round(satelliteIdx + self.f / self.planeCount)) % self.satsPerPlane
                simpleAdjacencyMatrix[idx, blockIdx + shiftIdxFrwd] = 1
                simpleAdjacencyMatrix[idx, blockIdx + shiftIdxBkwd] = 1
                simpleAdjacencyMatrix[idx, shiftPlaneIdxRight * self.satsPerPlane + shiftSatelliteIdxRight] = 1
                simpleAdjacencyMatrix[idx, shiftPlaneIdxLeft * self.satsPerPlane + shiftSatelliteIdxLeft] = 1
        return simpleAdjacencyMatrix
        # Same ideas could be used here to compact matrix

    def findShortestPath(self, adjacencyMatrix, startSatIdx, endSatIdx):
        if startSatIdx == endSatIdx:
            return [endSatIdx], 0

        visitedSats = np.zeros(self.getTotalSatCount(), dtype=bool)
        predSats = np.full(self.getTotalSatCount(), -1, dtype=int)

        queue = deque([startSatIdx])
        visitedSats[startSatIdx] = True

        while queue:
            currSatIdx = queue.popleft()
            if currSatIdx == endSatIdx:
                break
            #  BFS
            neighbours = np.where(adjacencyMatrix[currSatIdx])[0]
            for neighbour in neighbours:
                if not visitedSats[neighbour]:
                    visitedSats[neighbour] = True
                    predSats[neighbour] = currSatIdx
                    queue.append(neighbour)

        if not visitedSats[endSatIdx]:
            return None, np.inf

        pathStartToEnd = []
        step = endSatIdx
        while step != -1:
            pathStartToEnd.append(step)
            step = predSats[step]
        pathStartToEnd.reverse()
        pathLen = len(pathStartToEnd) - 1

        return pathStartToEnd, pathLen

class Constellation:

    def __init__(self, nameCode):
        self.totalSatCount = 0
        self.groups   = []
        self.elements = []
        self.stateEci = []
        self.loadFromConfig(nameCode)

    def loadFromConfig(self, nameCode):
        f = open('constellationsTest.json')
        jsonData = json.loads(f.read())

        for entryIdx in range(len(jsonData)):
            if (jsonData[entryIdx]['name']).lower() == nameCode.lower():
                print("Загружена группировка " + nameCode)
                constellationData = jsonData[entryIdx]

                for groupIdx in range(len(constellationData['Walkers'])):
                    self.groups.append(WalkerGroup(*constellationData['Walkers'][groupIdx]))
                    self.totalSatCount += self.groups[groupIdx].getTotalSatCount()

                f.close()
                return

        f.close()
        raise Exception('Группировка не найдена в файле')

    def getInitialState(self):
        self.elements = np.zeros((self.totalSatCount, 6))
        shift = 0

        for singleGroup in self.groups:
            ending = shift + singleGroup.getTotalSatCount()
            self.elements[shift:ending, :] = singleGroup.getInitialElements()
            shift = ending

    def propagateJ2(self, epochs):
        self.stateEci = np.zeros((self.totalSatCount, 3, len(epochs)))

        inclination = self.elements[:, 4]
        sma = self.elements[:, 0]
        raan0 = self.elements[:, 3]
        aol0 = self.elements[:, 5]

        raanPrecessionRate = -1.5 * (Const.earthJ2 * np.sqrt(Const.earthGM) * Const.earthRadius**2) \
                           / (sma**(7/2)) * np.cos(inclination)

        draconicOmega      = np.sqrt(Const.earthGM / sma**3) \
                           * (1 - 1.5 * Const.earthJ2 * (Const.earthRadius / sma)**2) \
                           * (1 - 4 * np.cos(inclination)**2)

        for epoch in epochs:
            aol = aol0 + epoch * draconicOmega
            raanOmega = raan0 + epoch * raanPrecessionRate

            epochState = sma * [
                (np.cos(aol) * np.cos(raanOmega) - np.sin(aol) * np.cos(inclination) * np.sin(raanOmega)),
                (np.cos(aol) * np.sin(raanOmega) + np.sin(aol) * np.cos(inclination) * np.cos(raanOmega)),
                (np.sin(aol) * np.sin(inclination))]

            self.stateEci[:, :, epochs.index(epoch)] = np.array(epochState).T

    def buildAdjacencyMatrixInnerPlaneLinksOnly(self, groupHeight):
        for singleGroup in self.groups:
            if singleGroup.altitude == groupHeight:
                return singleGroup.buildAdjacencyMatrixInnerPlaneLinksOnly()
    
    def buildAdjacencyMatrixNeighbourPlanesLinksIncluded(self, groupHeight):
        for singleGroup in self.groups:
            if singleGroup.altitude == groupHeight:
                return singleGroup.buildAdjacencyMatrixNeighbourPlanesLinksIncluded()

    def findShortestPathWithAdjacencyMatrix(self, groupHeight, startIdx, endIdx):
        for singleGroup in self.groups:
            if singleGroup.altitude == groupHeight:
                adjacencyMatrix = singleGroup.buildAdjacencyMatrixNeighbourPlanesLinksIncluded()
                return singleGroup.findShortestPath(adjacencyMatrix, startIdx, endIdx)
