import json
import os
from typing import Generic, List, NamedTuple, TypeVar

import numpy as np

T = TypeVar("T")


class Constant(Generic[T]):
    def __init__(self, value: T):
        self._value = value

    def __get__(self, instance, owner) -> T:
        return self._value

    def __set__(self, instance, value: T):
        raise AttributeError("Невозможно изменить константу")


class Earth(object):
    """Параметры (константы) планеты Земля"""

    radius: float = Constant(6378135)
    """Экваториальный радиус Земли [м]"""

    gm: float = Constant(3.986004415e14)
    """Гравитационный параметр Земли [м3/с2]"""

    j2: float = Constant(1.082626e-3)
    """Вторая зональная гармоника геопотенциала"""


class Walker(NamedTuple):
    """Базовый класс спутниковой группы типа Walker delta. Хранит неизменяемую информацию"""

    inclination: float
    """Наклонение орбиты [градусы]"""

    satsPerPlane: int
    """Число КА в каждой орбитальной плоскости группы"""

    planeCount: int
    """Число орбитальных плоскостей в группе"""

    phaseShift: int
    """Фазовый сдвиг по аргументу широты между КА в соседних плоскостях"""

    altitude: float
    """Высота орбиты [км]"""

    maxRaan: float
    """Максимум прямого восхождения восходящего узла (при распределении орбитальных плоскостей) [градусы]"""

    startRaan: float
    """Прямое восхождение восходящего узла (RAAN - Right Ascension of Ascending Node) для первой плоскости [градусы]"""


class WalkerGroup(Walker):
    """Класс, хранящий информацию о спутниковой группе типа Walker delta"""

    def totalSatCount(self) -> int:
        """Общее число КА в группе"""

        return self.satsPerPlane * self.planeCount

    def calcInitialElements(self) -> np.ndarray:
        """
        Выполняет расчёт Кеплеровых элементов орбиты группы спутников

        Returns:
            np.ndarray: матрица размера [satCount, 6], строки которой имеют вид

                [semiMajorAxis, 0, 0, raan, inclination, argumentOfLatitude]
        """

        startRaan: float = np.deg2rad(self.startRaan)
        maxRaan: float = np.deg2rad(self.maxRaan)
        inclination: float = np.deg2rad(self.inclination)
        altitude: float = self.altitude * 1000.0
        satCount: int = self.totalSatCount()

        raans = np.linspace(startRaan, startRaan + maxRaan, self.planeCount + 1)
        raans = raans[:-1] % (2 * np.pi)

        # sma - Semi-Major Axis
        sma = Earth.radius + altitude

        # aol - Argument of latitude
        aolStep = 2 * np.pi / self.satsPerPlane

        elements = np.zeros((satCount, 6))
        idx = 0

        for raanIdx, raan in enumerate(raans):
            aolShift = 2 * np.pi * self.phaseShift * raanIdx / satCount
            for satIdx in range(self.satsPerPlane):
                aol = aolStep * satIdx + aolShift

                elements[idx, :] = [sma, 0, 0, raan, inclination, aol]
                idx += 1

        return elements


class Constellation:
    """
    Класс, хранящий информацию о спутниковой группировке (созвездии)
    """

    def __init__(self, name: str):
        self.totalSatCount: int = 0
        """Общее количество КА группировки (созвездия)"""

        self.groups: List[WalkerGroup] = []
        """Спутниковые группы (типа Walker delta), входящие в группировку (созвездие)"""

        self.elements: np.ndarray = np.zeros(0)
        """Кеплеровы элементы орбиты всех КА группировки (созвездия)"""

        self.stateEci: np.ndarray = np.zeros(0)
        """Координаты всех КА группировки (созвездия) в инерциальной системе координат ECI (Earth-centered inertial)"""

        self.loadFromConfig(name)

    def loadFromConfig(self, name: str):
        """
        Загружает данные указанной спутниковой группировки из внешнего файла

        Args:
            name (str): имя группировки для загрузки

        Raises:
            Exception: если группировка с указанным именем не найдена в json-файле
        """

        scriptPath = os.path.dirname(__file__)
        jsonPath = os.path.realpath(f"{scriptPath}/../ConstellationsTest.json")
        with open(jsonPath) as f:
            jsonData = json.loads(f.read())

            for constellationData in jsonData:
                if (constellationData["name"]).lower() == name.lower():
                    print("Загружена группировка " + name)

                    for groupData in constellationData["Walkers"]:
                        self.groups.append(WalkerGroup(*groupData))
                        self.totalSatCount += self.groups[-1].totalSatCount()
                    return

        raise Exception("Группировка не найдена в файле")

    def calcInitialState(self):
        """
        Выполняет расчёт Кеплеровых элементы орбиты группировки (всех
        входящих в неё групп спутников)
        """

        self.elements = np.zeros((self.totalSatCount, 6))

        startIdx = 0
        for singleGroup in self.groups:
            endIdx = startIdx + singleGroup.totalSatCount()
            self.elements[startIdx:endIdx, :] = singleGroup.calcInitialElements()
            startIdx = endIdx

    def propagateJ2(self, epochs: List[int]):
        """
        Для всех указанных эпох расчитывает координаты КА в инерциальной системе
        координат ECI (Earth-centered inertial). Результаты работы сохраняются в
        поле `self.stateEci`

        Args:
            epochs (List[int]): список эпох для расчета координат
        """

        assert len(epochs) > 0
        epochs = np.array(epochs).reshape(1, -1)

        self.stateEci = np.zeros((self.totalSatCount, 3, len(epochs)))

        inclination = self.elements[:, 4]
        sma = self.elements[:, 0]
        raan0 = self.elements[:, 3]
        aol0 = self.elements[:, 5]

        inclinationСos = np.cos(inclination)
        inclinationSin = np.sin(inclination)

        earthJ2Radius = -1.5 * Earth.j2 * Earth.radius**2

        raanPrecessionRate = inclinationСos / sma ** (7 / 2)
        raanPrecessionRate *= earthJ2Radius * np.sqrt(Earth.gm)

        draconicOmega = 1.0 - 4.0 * inclinationСos**2
        draconicOmega *= 1.0 + earthJ2Radius / sma**2
        draconicOmega *= np.sqrt(Earth.gm / sma**3)

        raanOmega = raan0 + epochs.T * raanPrecessionRate
        raanOmegaCos = np.cos(raanOmega)
        raanOmegaSin = np.sin(raanOmega)

        aol = aol0 + epochs.T * draconicOmega
        aolCos = np.cos(aol)
        aolSin = np.sin(aol)

        stateEciX = aolCos * raanOmegaCos - aolSin * raanOmegaSin * inclinationСos
        stateEciY = aolCos * raanOmegaSin + aolSin * raanOmegaCos * inclinationСos
        stateEciZ = aolSin * inclinationSin

        self.stateEci = sma * [stateEciX, stateEciY, stateEciZ]
        self.stateEci = np.transpose(self.stateEci, [2, 0, 1])
