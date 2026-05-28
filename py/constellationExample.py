import random
from random import randint

import numpy as np
from Constellation import Constellation


def main():
    # Инициализация генератора случаных чисел для повторяемости результатов
    np.random.seed(0)
    random.seed(0)

    # Создание объекта типа Constellation, инициализация параметрами группировки Stalink из конфига
    constellation = Constellation("Starlink")

    # Вычисление элементов орбиты для всех КА в начальный момент
    constellation.calcInitialState()

    # Определение точек на оси времени, в которые будут проихзводиться расчёты
    maxEpoch = 3600
    epochs = list(range(maxEpoch))

    # Расчёт положений всех КА в заданные моменты времени
    constellation.propagateJ2(epochs)

    # Координаты случайного КА (в инерциальных осях) после этого можно прочитать из constellation.stateEci
    randomSatIdx = randint(0, constellation.totalSatCount - 1)
    randomEpochIdx = randint(0, len(epochs) - 1)
    print(f"Положение КА-{randomSatIdx} на эпоху {epochs[randomEpochIdx]}:")
    print(constellation.stateEci[randomSatIdx, :, randomEpochIdx])


if __name__ == "__main__":
    main()
