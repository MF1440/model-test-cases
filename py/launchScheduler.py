import numpy as np
import math
from npy_append_array import NpyAppendArray

"""
    LaunchScheduler - Класс который назначает новую посследовательность запусков,
    при которой минимальна разница между самым поздним временем выведения и самым ранним временем запуска.

    Алгоритм:
        1) Разбиваем входную матрицу запусков на батчи.
        2) Для каждого батча считаем вспомогательный вектор, который хранит разницу между первым 
            и последним индексами орбит для которых кол-во выводимых спутников больше нуля. Каждый такой вектор
            добавляем в общий вектор с сохранением порядкового номера.
        3) Сортируем получившийся вектор в порядке возрастания разницы индексов орбит. Получаем новый порядок запусков,
            который имеет минимальную разницу между самым поздним временем выведения и самым ранним временем запуска.
    Сложность алгоритма O(n*m + n*log(n)), где n - кол-во запусков, m - кол-во орбит.
"""

class LaunchScheduler:
    """
        inScheduleMatrix - Путь к файлу с матрицей запусков.
        outScheduleMatrix - Путь к файлу куда сохранить новую матрицу запусков.
        launchsNum - Кол-во запусков.
        orbitsNum - Кол-во орбит на которые будут выводиться спутники.
        maxMemUseBytes - Максимальное кол-во допустимой памяти.
    """
    def __init__(self, inScheduleMatrix, outScheduleMatrix, launchsNum, orbitsNum, maxMemUseBytes):
        self.inScheduleMatrix = inScheduleMatrix
        self.outScheduleMatrix = outScheduleMatrix
        self.launchsNum = launchsNum
        self.orbitsNum = orbitsNum
        self.maxMemUseBytes = maxMemUseBytes

        self.batchSize = math.floor(self.maxMemUseBytes / (self.orbitsNum))
        self.batchsNum = math.ceil(self.launchsNum / self.batchSize)
        self.launchDiff = []
    
    """
        Для текущего запуска вернуть индекс первой орбиты для которой кол-во выводимых
        спутников больше нуля
    """
    def firstNonZero(self, launchTable):
        for i, sNum in enumerate(launchTable):
            if sNum != 0:
                return i
        return -1

    """
        Для текущего запуска вернуть индекс последней орбиты для которой кол-во выводимых
        спутников больше нуля
    """
    def lastNonZero(self, launchTable):
        for i, sNum in reversed(list(enumerate(launchTable))):
            if sNum != 0:
                return i
        return -1

    """
        Подсчет разницы между первым 
        и последним индексами орбит для которых кол-во выводимых спутников больше нуля.
    """
    def calcLaunchDiff(self, scheduleMatrix):
        newLaunchDiff = [self.lastNonZero(l) - self.firstNonZero(l) for l in scheduleMatrix]
        self.launchDiff.extend(newLaunchDiff)

    """
        Сортировка подсчитанного вектора разниц
    """
    def sortTotalLaunchDiff(self):
        enumLaunchDiff = list(enumerate(self.launchDiff))
        enumLaunchDiff.sort(key=lambda item: item[1], reverse=True)
        newOrder = [l[0] for l in enumLaunchDiff]
        return newOrder
    
    """
        Сохранение новой посследовательности запусков в файл.
    """
    def saveNewSchedule(self, newOrder):
        oldScheduleMatrixMemMap = np.load(self.inScheduleMatrix, mmap_mode='r')
        naa = NpyAppendArray(self.outScheduleMatrix, delete_if_exists=True)
        for i in range(self.batchsNum):
            startId = i * self.batchSize
            endId = (i + 1) * self.batchSize
            newScheduleMatrix = oldScheduleMatrixMemMap[newOrder[startId:endId]]
            naa.append(newScheduleMatrix)
        
    """
        Основная функция, которая находит новую посследовательность запусков.
        Обрабатывает входную матрица запусков в несколько шагов, чтобы уложиться
        в ограничения по памяти. 
    """
    def solve(self):
        scheduleMatrixMemMap = np.load(self.inScheduleMatrix, mmap_mode='r')
        for i in range(self.batchsNum):
            print("process batch #" +  str(i + 1))
            startRow = i * self.batchSize
            endRow = (i + 1) * self.batchSize
            scheduleMatrix = scheduleMatrixMemMap[startRow:endRow]
            self.calcLaunchDiff(scheduleMatrix)

        newOrder = self.sortTotalLaunchDiff()
        self.saveNewSchedule(newOrder)
    

