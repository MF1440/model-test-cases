#!/usr/bin/env python
# coding: utf-8

# Реализация задания #3:
#   В файле gatewaysTest.json заданы координаты шлюзовых станций, расположенных на Земле.
#   Реализуйте вычислительную процедуру, позволяющую для набора произвольных моментов времени
#   определить, какие КА находятся в зоне видимости шлюзовых станций
#   (группировку использовать из файла constellationTest.json).
#   Условие нахождения КА в зоне видимости шлюзовой станции – угол места ε ≥ 25°
#   (ограничения на α не накладываются).
#
# Решение:
#   Синус угла места равен скалярному произведению внешней нормали к поверхности Земли
#   на единичный вектор, направленный из местоположения шлюзовой станции к КА.
#   Условие видимости:
#       n * (r_sat - r_station) >= sin(25 / 180 * pi)
#   Координаты шлюзовой станции вычислим через заданные геодезические координаты и время.


import json

import numpy as np
from math import sin, cos, pi

from constellation import Const, Constellation


def geodetic2cartesian(latitude, longtitude, altitude):
    '''
    Возвращает Евклидовы координаты точки, заданной в геодезических координатах.
    Для выходных данных используется система координат, вращающаяся вместе с Землей.
    Земля считается идеальной сферой с центром в начале координат.
    Нулевой меридиан принадлежит плоскости Oxz.
    '''
    R = Const.earthRadius + altitude
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longtitude)
    return np.stack([
        R * np.cos(lat_rad) * np.cos(lon_rad),
        R * np.cos(lat_rad) * np.sin(lon_rad),
        R * np.sin(lat_rad),
    ], axis=1)


def loadGatewaysFromJson(fileName):
    '''
    Загружает местоположение шлюзовых станций из файла fileName
    '''
    minElevationAngle = 25

    with open(fileName, 'r') as fIn:
        gatewayData = json.load(fIn)
        try:
            res = np.array([
                [gw['lat'], gw['lon'], gw['altitude'], minElevationAngle,] \
                    for gw in gatewayData
            ])
            return res
        except (KeyError, ValueError):
            raise Exception(f'Wrong gateway data in \'{fileName}\'')


def loadEpochsFromJson(fileName):
    '''
    Загружает указанные моменты времени из файла fileName
    '''
    with open(fileName, 'r') as fIn:
        epochsData = json.load(fIn)
        try:
            return [float(epoch) for epoch in epochsData]
        except (KeyError, ValueError):
            raise Exception(f'Wrong epochs data in \'{fileName}\'')


def predictSatelliteElevation(gateways, constellation, epochs):
    '''
    Определить углы места спутников для каждой шлюзовой станции
    для каждого указанного момента времени.
    Возвращает двумерный массив (индекс_спутника, индекс_шлюзовой_станции, индекс_эпохи)
    '''
    res = np.zeros((constellation.totalSatCount, gateways.shape[0], len(epochs)))

    # Декартовы координаты во вращающейся системе координат, привязанной к Земле
    gatewaysGsk = geodetic2cartesian(gateways[:,0], gateways[:,1], gateways[:,2])

    satellitesPos = constellation.predictCoordinates(epochs)

    for epochIdx, epoch in enumerate(epochs):
        # Матрица поворота для определения координат шлюзовых станций в инерциальной СК
        angle = epoch * Const.earthOmega
        rotateMatrix = np.array(
            [[cos(angle), -sin(angle), 0],
             [sin(angle), cos(angle), 0],
             [0, 0, 1]]
        )

        for gatewayIdx, gatewayGsk in enumerate(gatewaysGsk):
            # Координаты шлюзовой станции в инерциальной СК
            gatewayIsk = rotateMatrix.dot(gatewayGsk)
            # Поскольку Земля принята идеальной сферой, нормаль к Земле есть
            # нормированный радиус-вектор от центра Земли к шлюзовой станции
            gatewayIskNorm = gatewayIsk / np.linalg.norm(gatewayIsk)

            for satIdx in range(satellitesPos.shape[0]):
                satPos = satellitesPos[satIdx, :, epochIdx]
                gw2sat = satPos - gatewayIsk
                gw2satDirection = gw2sat / np.linalg.norm(gw2sat)

                res[satIdx, gatewayIdx, epochIdx] = gatewayIskNorm.dot(gw2satDirection)

    return res


def findVisibleSatellites(gateways, constellation, epochs):
    '''
    Возвращает перечень видимых КА для каждой шлюзовой станции и каждого момента времени
    '''
    elevation = predictSatelliteElevation(gateways, constellation, epochs)
    # Факт видимости шлюзовых станций
    visibility = np.stack([
        (elevation[:, gatewayIdx, :] >= sin(np.deg2rad(gateways[gatewayIdx, 3]))) \
            for gatewayIdx in range(elevation.shape[1])], axis=1)
    # Перечень видимых спутников для каждой шлюзовой станции
    visibleSats = [[[satIdx \
        for satIdx, visible in enumerate(visibility[:, gatewayIdx, epochIdx]) if visible] \
            for gatewayIdx in range(visibility.shape[1])] \
                for epochIdx in range(visibility.shape[2])]
    return visibleSats

def main():

    import argparse
    parser = argparse.ArgumentParser(description='Задание 3. Расчет видимости спутников')
    parser.add_argument('-c', '--constellation', default='constellationsTest.json',
        help='JSON файл, содержащий параметры орбит спутников')
    parser.add_argument('-n', '--constellation-name', default='Starlink',
        help='Имя спутниковой группировки в JSON файле')
    parser.add_argument('-g', '--gateways', default='gatewaysTest.json',
        help='JSON файл, содержащий местоположение шлюзовых станций')
    parser.add_argument('-e', '--epochs', default=None,
        help='JSON файл, содержащий список моментов времени для расчета видимости')
    parser.add_argument('-o', '--output-file', default=None,
        help='Имя выходного файла. stdout если не указано.')
    args = parser.parse_args()

    gateways = loadGatewaysFromJson(args.gateways)
    constellation = Constellation(args.constellation, 'Starlink')
    constellation.updateInitialState()

    if args.epochs:
        epochs = loadEpochsFromJson(args.epochs)
    else:
        epochs = list(range(0, 10001, 100))

    visibleSats = findVisibleSatellites(gateways, constellation, epochs)

    # Выберем случайный момент времени и случайную шлюзовую станцию
    from random import randint
    epochIdx = randint(0, len(epochs) - 1)
    gatewayIdx = randint(0, len(gateways) - 1)
    # Список всех видимых спутников
    print(f'Наблюдаемые КА для шлюзовой станции {gatewayIdx} в момент времени {epochs[epochIdx]}:')
    print(', '.join(str(x) for x in visibleSats[epochIdx][gatewayIdx]))


if __name__ == '__main__':
    main()
