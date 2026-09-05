from constellation import *
from random import randint

# создание объекта типа Constellation, инициализация параметрами группировки Stalink из конфига
constellation = Constellation('Starlink')

# вычисление элементов орбиты для всех КА в начальный момент
constellation.getInitialState()

# usage example
print(constellation.buildAdjacencyMatrixInnerPlaneLinksOnly(720))
print(constellation.buildAdjacencyMatrixNeighbourPlanesLinksIncluded(720))
print(constellation.findShortestPathWithAdjacencyMatrix(720, 3, 9))

# определение точек на оси времени, в которые будут проихзводиться расчёты
epochs = list(range(1002))

# расчёт положений всех КА в заданные моменты времени
constellation.propagateJ2(epochs)

# Координаты случайного КА (в инерциальных осях) после этого можно прочитать из constellation.stateEci
satIdx = randint(0, constellation.totalSatCount - 1)
epochIdx = randint(0, len(epochs) - 1)
print("Положение КА-" + str(satIdx) + " на эпоху " + str(epochs[epochIdx]) + ":")
print(constellation.stateEci[satIdx, :, epochIdx])
