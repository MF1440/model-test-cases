clc
clear

tic

%создание объекта типа Constellation, инициализация параметрами группировки Stalink из конфига
constellation = Constellation('Starlink');

%вычисление элементов орбиты для всех КА в начальный момент
constellation.calcInitialState();

% определение точек на оси времени, в которые будут производиться расчёты
epochs = (0: 1000: 6000);

% расчёт положений всех КА в заданные моменты времени
constellation.propagateJ2(epochs);

% Координаты случайного КА (в инерциальных осях) после этого можно прочитать из constellation.state.eci
satIdx = ceil(constellation.totalSatCount * rand());
epochIdx = ceil(length(epochs) * rand());

disp(['Положение КА-' num2str(satIdx) ' на эпоху ' num2str(epochs(epochIdx)) ':']);
disp(constellation.state.eci(satIdx, :, epochIdx));

% задание 2
aol = max(constellation.state.elements(:, 6));
checkGlobalView(constellation.groupArray{1}, aol, alpha) %проверка глобальной видимости Земли первого массива КА группировки в зависимости от угла обзора alpha (задается вводом)
checkGlobalView(constellation.groupArray{2}, aol, alpha) %проверка глобальной видимости второго Земли массива КА группировки в зависимости от угла обзора alpha (задается вводом)
disp(['Минимальный угол обзора для группировки 1=', num2str(checkMinAlphaGlobalView(constellation.groupArray{1}, aol))]); %минимально возможный угол обзора для выполнения условия глобальной видимости Земли
disp(['Минимальный угол обзора для группировки 2=', num2str(checkMinAlphaGlobalView(constellation.groupArray{2}, aol))]); %минимально возможный угол обзора для выполнения условия глобальной видимости Земли

toc
