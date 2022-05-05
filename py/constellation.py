#!/usr/bin/env python
# coding: utf-8


# TODO:
# - Проверить вычисление координат КА. Похоже, что предыдущая реализация не учитывала
#   указанную долготу восходящего узла. См. воркэраунд в коде.
# - Реализовать юнит тесты для core-функционала
# - Для класса Constellation разделить core-функционал и ввод-вывод (загрузку из JSON)
# - Рассмотреть возможность изменения API: заменить массив с описанием группы
#   спутников на объект. Например:
#       "Walkers": [ { "inclination": 53, "satsPerPlane": 50, ... }, ... ]
# - Избавиться от конструкций, специфичных для Python2
# - Использовать библиотеку logging


import numpy as np
import json
from typing import NamedTuple


class Parameters(object):
    pass

Const = Parameters()
Const.earthRadius = 6378135;           # Экваториальный радиус Земли [km]
Const.earthGM = 3.986004415e+14;       # Гравитационный параметр Земли [m3/s2]
Const.earthJ2 = 1.082626e-3;           # Второй динамический коэффициент формы Земли
Const.earthOmega = 7.2921158553e-5     # Угловая скорость вращения Земли [1/s]

group = Parameters()


class Walker(NamedTuple):
    inclination:  float     # наклонение
    satsPerPlane: int       # количество спутников в одной орбитальной плоскости
    planeCount:   int       # количество орбитальных плоскостей
    f:            int       # истинная аномалия
    altitude:     float     # высота орбиты над поверхностью Земли
    maxRaan:      float     # долгота восходящего узла "последней" орбитальной плоскости
    startRaan:    float     # долгота восходящего узла "первой" орбитальной плоскости

class WalkerGroup(Walker):
    '''
    Группа спутников, находящихся в орбитальных плоскостях с одинаковым наклонением
    '''

    def getTotalSatCount(self):
        '''
        Возвращает количество спутников внутри группы
        '''
        return self.satsPerPlane * self.planeCount;

    def getInitialElements(self):
        '''
        Возвращает элементы орбиты всех спутников:
            numpy.array([
                [...],
                [sma_i, ?_i, ?_i, raan_i, inclination_i, aol_i], # Для i-го спутника
                [...]
            ])
        Здесь
            sma_i: большая полуось орбиты
            ?_i: эксцентриситет орбиты
            ?_i: аргумент перицентра
            raan_i: долгота восходящего узла
            inclination_i: наклонение орбиты
            aol_i: аномалия
        '''

        startRaan = np.deg2rad(self.startRaan)
        maxRaan = np.deg2rad(self.maxRaan)
        inclination = np.deg2rad(self.inclination)
        altitude = self.altitude * 1000

        raans = np.linspace(startRaan, startRaan + maxRaan, self.planeCount + 1)
        raans = raans[:-1] % (2 * np.pi)

        elements = np.zeros((self.getTotalSatCount(), 6))
        idx = 0
        for raanIdx, raan in enumerate(raans):
            for satIdx in range(self.satsPerPlane):
                sma = Const.earthRadius + altitude
                aol = 2 * np.pi / self.satsPerPlane * (satIdx + 1) \
                    + 2 * np.pi / self.getTotalSatCount() * self.f * raanIdx

                elements[idx, :] = [sma, 0, 0, raan, inclination, aol]
                idx += 1

        return elements


class Constellation:
    '''
    Параметры группировки спутников
    '''

    def __init__(self, fileName, nameCode):
        self.totalSatCount = 0
        self.groups = []
        self.elements = []
        self.loadFromConfig(fileName, nameCode)

    def loadFromConfig(self, fileName, nameCode):
        '''
        Загрузить параметры группировки с именем nameCode из файла fileName
        '''
        # TODO: Реализовать обработку ошибок ввода/вывода при чтении файла

        with open(fileName) as inputFile:
            jsonData = json.load(inputFile)

            for constellationData in jsonData:
                if (constellationData['name']).lower() == nameCode.lower():
                    print("Загружена группировка " + nameCode)

                    for groupData in constellationData['Walkers']:
                        group = WalkerGroup(*groupData)
                        self.groups.append(group)
                        self.totalSatCount += group.getTotalSatCount()

                    return

            raise Exception('Группировка не найдена в файле')

    def updateInitialState(self):
        '''
        Заполняет атрибут .elements элементами орбиты всех спутников
        '''
        self.elements = np.concatenate(tuple(group.getInitialElements() \
            for group in self.groups))


    def predictCoordinates(self, epochs):
        '''
        Возвращает координаты спутников для заданных эпох epochs
        '''
        res = np.zeros((self.totalSatCount, 3, len(epochs)))

        inclination = self.elements[:, 4]
        sma         = self.elements[:, 0]
        Omega0      = np.sqrt(Const.earthGM / sma**3)
        aol0        = self.elements[:, 5]

        sin_raan    = np.sin(self.elements[:, 3])
        cos_raan    = np.cos(self.elements[:, 3])

        # TODO: Дополнить формулу для случая ненулевого эксцентриситета
        raanPrecessionRate = -1.5 * (Const.earthJ2 * np.sqrt(Const.earthGM) * Const.earthRadius**2) \
            / (sma**(7/2)) * np.cos(inclination)
        draconicOmega      = np.sqrt(Const.earthGM / sma**3) \
            * (1 - 1.5 * Const.earthJ2 * (Const.earthRadius / sma)**2) \
            * (1 - 4 * np.cos(inclination)**2)

        breakpoint

        for epochIdx, epoch in enumerate(epochs):
            aol = aol0 + epoch * draconicOmega
            Omega = Omega0 + epoch * raanPrecessionRate

            epochState = sma * [(np.cos(aol) * np.cos(Omega) - np.sin(aol) * np.cos(inclination) * np.sin(Omega)),
                                (np.cos(aol) * np.sin(Omega) + np.sin(aol) * np.cos(inclination) * np.cos(Omega)),
                                (np.sin(aol) * np.sin(inclination))]

            # Поворот на долготу восходящего узла (воркэраунд)
            # TODO: объединить с предыдущей строкой в единую формулу
            epochState = [
                epochState[0] * cos_raan - epochState[1] * sin_raan,
                epochState[0] * sin_raan + epochState[1] * cos_raan,
                epochState[2],
            ]

            res[:, :, epochIdx]  = np.array(epochState).T

        return res


    def propagateJ2(self, epochs):
        '''
        Заполняет атрибут .stateEci координатами спутников для заданных эпох epochs
        '''
        self.stateEci = self.predictCoordinates(epochs)

