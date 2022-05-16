function coveredSats = findVisibleSats(constellation, fileName, epochList, epochIdx, elevAngleMinDeg)

    % ОПИСАНИЕ:
    % Возвращает массив индексов спутников, видимых с конкретной шлюзовой станции.
    % constellation - объект класса Constellation 
    % fileName - имя файла с координатами расположения шлюзовых станций
    % epochList - массив точек на оси времени, когда проводятся расчеты 
    % epochIdx - индекс точки в массиве моментов расчета траектории
    % спутников
    % elevAngleMinDeg - минимальный угол места, при котором КА находится в
    % зоне видимости шлюзовой станции [град.]
    %
    % ВХОДНЫЕ ДАННЫЕ:
    % constellation - объект класса Constellation
    % fileName - символьная строка
    % epochList - одномерный массив скаляров
    % epochIdx - целое положительное число, значение которого находится в диапазоне [1, length(epochList)]  
    % elevAngleMinDeg - число от 0 до 90 
    %
    % ВЫХОДНЫЕ ЗНАЧЕНИЯ:
    % coveredSats - двумерный массив ячеек, где каждая ячейка - массив индексов спутников, видимых с конкретной шлюзовой станции в данный момент времени.

    epoch = epochList(epochIdx);
    
    stationList = jsondecode(fileread(fileName));
    
    coveredSats = {};
    
    satEciList = constellation.state.eci(:, :, epochIdx);
    
    stationEciList = zeros(length(stationList), 3);
    
    % Вычисление координат станций в системе ECI
    for stationIdx = 1: length(stationList)
        stationEciList(stationIdx,:) = calcEci(stationList(stationIdx).lat, stationList(stationIdx).lon, stationList(stationIdx).altitude, epoch);
    end
    
    for stationIdx = 1: length(stationEciList)

        visibleSatsPerStation = [];

        for satelliteIdx = 1: length(satEciList)

            stationSatVec = satEciList(satelliteIdx,:) - stationEciList(stationIdx,:);
            elevAngleRad = pi/2 - abs(calcAngle(stationSatVec, stationEciList(stationIdx,:)));
            
            % Проверка условия нахождения КА в зоне видимости шлюзовой станции
            if  rad2deg(elevAngleRad) >= elevAngleMinDeg
                visibleSatsPerStation(end+1) = satelliteIdx;
            end

        end

        coveredSats{end+1} = visibleSatsPerStation;
    end

