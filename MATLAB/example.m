clc
clear 
tic
%создание объекта типа Constellation, инициализация параметрами группировки Stalink из конфига
constellation = Constellation('Starlink');

%вычисление элементов орбиты для всех КА в начальный момент
constellation.updateInitialState();

% определение точек на оси времени, в которые будут производиться расчёты
epochList = (0: 1000: 6000);

% расчёт положений всех КА в заданные моменты времени
constellation.propagateJ2(epochList);

% Координаты случайного КА (в инерциальных осях) после этого можно прочитать из constellation.state.eci
satIdx = ceil(constellation.totalSatCount * rand());
epochIdx = ceil(length(epochList) * rand());

disp(['Положение КА-' num2str(satIdx) ' на эпоху ' num2str(epochList(epochIdx)) ':']);
disp(constellation.state.eci(satIdx, :, epochIdx));


% Задание 3

% Минимальный угол места спутника для нахождения в зоне видимости шлюзовой станции
elevAngleMinDeg = 25;

% Файл с таблицей координат шлюзовых станций
filename = 'gatewaysTest.json';

stationTable = findVisibleSats(constellation, filename, epochList, epochIdx, elevAngleMinDeg);
for stationIdx = 1: length(stationTable)
    disp(['Спутники, наблюдаемые станцией № ' num2str(stationIdx) ' на эпоху ' num2str(epochList(epochIdx)) ':']);
    disp(stationTable(stationIdx))
end
toc